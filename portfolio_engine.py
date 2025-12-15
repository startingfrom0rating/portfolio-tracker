
import pandas as pd
import yfinance as yf
import re
import os

class PortfolioEngine:
    def __init__(self, transaction_file, open_position_file=None):
        self.transaction_file = transaction_file
        self.open_position_file = open_position_file
        self.transactions = None
        self.positions = None
        self.holdings = {} # {ticker: quantity}
        self.avg_costs = {} # {ticker: avg_cost_per_share_usd}
        self.cost_basis = {} # {ticker: total_invested_usd}
        self.cash_balance = 500000.0
        self.market_data = {}
        self.fallback_prices = {} # {symbol: price (USD)}
        self.errors = []
        self.FORCE_FALLBACK = ['HWC.L', 'HWC', 'B-T-6.250-15052030']

    def clean_currency(self, val):
        """Removes currency symbols and converts to float."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # Remove common currency symbols and commas
            clean = re.sub(r'[¥£$C,\s]', '', val)
            try:
                return float(clean)
            except ValueError:
                return 0.0
        return 0.0

    def get_yf_ticker(self, row):
        """Maps CSV symbol/exchange to Yahoo Finance ticker."""
        symbol = str(row['Symbol']).strip()
        exchange = str(row['Exchange']).strip()
        currency = str(row['Currency']).strip()

        # Logic based on observation
        if exchange == 'US':
            # Exception for HWC (Highway Capital PLC) which is likely UK 
            # or if Currency is GBP.
            if currency == 'GBP':
                return f"{symbol}.L"
            return symbol
        elif exchange == 'T':
            return f"{symbol}.TO"
        elif currency == 'JPY' or exchange == 'PK':
             return f"{symbol}.T"
        elif currency == 'GBP': # Catch all GBP
             return f"{symbol}.L"
        # Fallback
        return symbol

    def load_data(self):
        """Loads and processes the transaction history."""
        try:
            df = pd.read_csv(self.transaction_file)
            
            # Clean numeric columns
            df['Quantity'] = df['Quantity'].apply(self.clean_currency)
            df['Amount'] = df['Amount'].apply(self.clean_currency)
            df['Price'] = df['Price'].apply(self.clean_currency)
            
            # Map Tickers
            df['YF_Ticker'] = df.apply(self.get_yf_ticker, axis=1)
            
            self.transactions = df
            
            # Process Cash and Holdings
            self._process_transactions()

            # Load Open Positions for Fallback Pricing
            if self.open_position_file and os.path.exists(self.open_position_file):
                try:
                    odf = pd.read_csv(self.open_position_file)
                    # Create map: Symbol -> LastPrice
                    # Note: OpenPosition prices might be in local currency?
                    # Check HWC: PricePaid 63.06 (GBP), LastPrice 63.28 (USD/GBP?).
                    # Check 8053: LastPrice 4900 (JPY).
                    # Check Bonds: LastPrice 1047.59...
                    # We need to be careful.
                    # If we use fallback, we assume the price in OpenPosition is usable.
                    # Best effort: Map Symbol -> (LastPrice, Currency)
                    # OpenPosition columns: Symbol, Quantity, Currency, LastPrice...
                    for _, row in odf.iterrows():
                        sym = str(row['Symbol']).strip()
                        price = float(str(row['LastPrice']).replace(',','').replace('$',''))
                        curr = str(row['Currency']).strip()
                        self.fallback_prices[sym] = {'Price': price, 'Currency': curr}
                        # Also map the YF ticker mapping to this symbol if possible
                        yf_tk = self.get_yf_ticker({'Symbol': sym, 'Exchange':'', 'Currency':curr})
                        if yf_tk != sym:
                            self.fallback_prices[yf_tk] = {'Price': price, 'Currency': curr}

                except Exception as e:
                    print(f"Warning loading open positions: {e}")
            
            return True
        except Exception as e:
            self.errors.append(f"Error loading data: {str(e)}")
            return False

    def _process_transactions(self):
        """Calculates current holdings and cash balance."""
        # Calculate Cash
        # Amount in CSV is negative for buys, positive for sells/divs (usually)
        # We assume Amount column is already in USD based on analysis of JPY/CAD rows.
        total_cash_flow = self.transactions['Amount'].sum()
        total_cash_flow = self.transactions['Amount'].sum()
        self.cash_balance = 500000.0 + total_cash_flow
        
        # Calculate Cost Basis
        self._calculate_cost_basis()
        
        # Calculate Holdings
        holdings = {}
        # Only Buy/Sell affects quantity?
        # Transaction Types: "Market - Buy", "Market - Sell", "Limit - Buy", etc.
        # "Dividends" -> Qty usually implies nothing or maybe DRIP?
        # Let's check Dividend rows in data.
        # Row 15: K, Dividends, Qty 308. Price ... Amount ...
        # Wait, if Dividend has Quantity 308, does it mean I got 308 shares?
        # Usually Dividend Quantity column in these reports reflects the shares held *at that time*, not new shares.
        # UNLESS it's a Reinvestment (DRIP).
        # Check Row 15: Amount C$10.76. Price C$0.04.
        # 308 * 0.04 = 12.32.
        # If it was 308 NEW shares, value would be huge.
        # So "Dividends" quantity is likely just informational (reference quantity).
        # We should only sum Quantity for "Buy" and "Sell" types.
        
        for idx, row in self.transactions.iterrows():
            txn_type = str(row['TransactionType']).lower()
            ticker = row['YF_Ticker']
            qty = row['Quantity']
            
            if 'buy' in txn_type or 'sell' in txn_type:
                # Note: Sells in this CSV have negative Quantity?
                # Let's check.
                # Row 25: MP, Limit - Sell, Quantity -100.
                # Row 2: ARI, Buy, Quantity 4.
                # So we can just Sum the Quantity column for Buy/Sell rows!
                # But wait, what if Dividends have quantity?
                # Row 15 K Dividends Qty 308.
                # If I sum that, I'll double count.
                # So I must filter by Type.
                if 'dividend' not in txn_type and 'payment' not in txn_type:
                     holdings[ticker] = holdings.get(ticker, 0.0) + qty
            
        # Filter out zero positions
        self.holdings = {k: v for k, v in holdings.items() if abs(v) > 0.001}
        
    def _calculate_cost_basis(self):
        """Calculates Weighted Average Cost Basis."""
        # Cost Basis = Sum(Buy Qty * Buy Price) / Total Buy Qty
        # NOTE: This is simplified. FIFO/LIFO is harder. WAC is standard for simple views.
        # Problem: We have "Amount" in USD.
        # Buys are negative amount. Cost = -Amount.
        # Sells reduce quantity but don't change Avg Cost per share (usually).
        # We need to iterate chronologically.
        
        costs = {} # {ticker: {'total_cost': 0.0, 'shares': 0.0}}
        
        # Sort
        df = self.transactions.sort_values('CreateDate')
        
        for idx, row in df.iterrows():
            txn_type = str(row['TransactionType']).lower()
            if 'buy' not in txn_type and 'sell' not in txn_type:
                continue
                
            ticker = row['YF_Ticker']
            qty = row['Quantity'] # Positive for Buy (usually), Check?
            # In this CSV, Buy Qty is Positive. Sell is Negative.
            # Row 2: Buy 4. Row 25: Sell -100.
            
            amt = row['Amount'] # Buy is Negative USD. Sell is Positive USD.
            
            if ticker not in costs:
                costs[ticker] = {'total_cost': 0.0, 'shares': 0.0}
            
            curr = costs[ticker]
            
            if qty > 0: # Buy
                # Add to cost
                # Cost is -Amount (Amount is neg)
                cost_usd = -amt
                curr['total_cost'] += cost_usd
                curr['shares'] += qty
            elif qty < 0: # Sell
                # Reduce shares. Cost per share stays same?
                # Realized PnL event.
                # Reduce total_cost proportionally to maintain Avg Cost.
                if curr['shares'] > 0:
                    avg_cost = curr['total_cost'] / curr['shares']
                    # Removed cost
                    removed_cost = avg_cost * abs(qty)
                    curr['total_cost'] -= removed_cost
                    curr['shares'] += qty # qty is negative
                else:
                    # Short selling?
                    pass
        
        # Result
        for tk, data in costs.items():
            if data['shares'] > 0.001:
                self.avg_costs[tk] = data['total_cost'] / data['shares']
                self.cost_basis[tk] = data['total_cost']
            else:
                self.avg_costs[tk] = 0.0
                self.cost_basis[tk] = 0.0


    def fetch_market_data(self):
        """Fetches current prices for all held tickers."""
        tickers = list(self.holdings.keys())
        if not tickers:
            return

        # Download batch
        # Using counters to handle potential issues
        print(f"Fetching data for: {tickers}")
        try:
            # We only need latest price for current valuation
            # But for history we might need more. For now, let's get 1mo to be safe, or 1d if just current.
            # Plan mentions historical equity curve, so we might need full history later.
            # For the MVP validation step, let's get '1d' or '5d' to ensure we have a close.
            # Actually, fetch history separately for the chart.
            # Here let's get current stats.
            
            # Note: Tickers with suffixes might need FX conversion handling if yfinance returns local price.
            # yfinance returns price in currency of exchange.
            # info['currency'] tells us.
            
            data = yf.Tickers(" ".join(tickers))
            
            metrics = []
            metrics = []
            for ticker in tickers:
                try:
                    # Check Force Fallback first
                    if ticker in self.FORCE_FALLBACK:
                         hist = pd.DataFrame() # Treat as empty to trigger fallback logic
                    else:
                        # Better to use history metadata for price to avoid 'info' broken endpoint issues
                        hist = data.tickers[ticker].history(period="1d")
                    
                    if hist.empty:
                        # Fallback
                        fallback = self.fallback_prices.get(ticker, self.fallback_prices.get(ticker.replace('.TO','').replace('.T','').replace('.L','')))
                        if fallback:
                            current_price = fallback['Price']
                            currency = fallback['Currency']
                        else:
                            current_price = 0.0
                            currency = 'USD'
                    else:
                        current_price = hist['Close'].iloc[-1]
                        currency = 'USD'
                        if ticker.endswith('.TO'):
                            currency = 'CAD'
                        elif ticker.endswith('.T'):
                            currency = 'JPY'
                        elif ticker.endswith('.L'):
                            currency = 'GBP' 
                    
                    metrics.append({
                        'Ticker': ticker,
                        'Price': current_price,
                        'Currency': currency
                    })
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
                    metrics.append({'Ticker': ticker, 'Price': 0.0, 'Currency': 'USD'})
            
            self.market_data = pd.DataFrame(metrics).set_index('Ticker')
            
            # Fetch FX
            # We need CADUSD=X, JPYUSD=X to convert TO USD.
            # Or CAD=X (USD/CAD).
            # Let's standardize on USD/YYY?
            # JPY=X -> Rate ~ 150 (USD/JPY). Value_USD = Value_JPY / Rate.
            # CAD=X -> Rate ~ 1.35 (USD/CAD). Value_USD = Value_CAD / Rate.
            # So Price_USD = Price_Local / Rate.
            
            fx_tickers = ['JPY=X', 'CAD=X'] # GBP=X if needed
            fx_data = yf.download(fx_tickers, period="1d", progress=False)['Close'].iloc[-1]
            
            self.fx_rates = {
                'JPY': fx_data['JPY=X'] if 'JPY=X' in fx_data else 1.0,
                'CAD': fx_data['CAD=X'] if 'CAD=X' in fx_data else 1.0,
                'USD': 1.0
            }
            # Handle Single ticker result (series vs df)
            if isinstance(fx_data, float): # Only one requested?
                # Not applicable here as we requested list.
                pass
                
        except Exception as e:
             self.errors.append(f"Error fetching market data: {str(e)}")

    def get_valuations(self):
        """Computes value per position and total."""
        results = []
        total_equity = 0.0
        
        for ticker, shares in self.holdings.items():
            if ticker in self.market_data.index:
                price = self.market_data.loc[ticker, 'Price']
                currency = self.market_data.loc[ticker, 'Currency']
                
                # FX Conversion
                fx_rate = self.fx_rates.get(currency, 1.0)
                # Assuming Quote is Local, FX is USD/Local (e.g. 150 JPY per USD).
                # So Price_USD = Price / FX.
                # NOTE: Ensure FX logic is correct.
                # JPY=X is 154.  1000 JPY / 154 = 6.49 USD. Correct.
                # CAD=X is 1.40. 100 CAD / 1.40 = 71 USD. Correct.
                
                market_val_local = shares * price
                market_val_usd = market_val_local / fx_rate
                
                results.append({
                    'Ticker': ticker,
                    'Net Shares': shares,
                    'Price (Local)': price,
                    'Currency': currency,
                    'Market Value (USD)': market_val_usd,
                    'Avg Cost': self.avg_costs.get(ticker, 0.0),
                    'Unrealized PnL': market_val_usd - (shares * self.avg_costs.get(ticker, 0.0)),
                    'PnL %': ((price / (self.avg_costs.get(ticker, 1.0) * fx_rate)) - 1) * 100 if self.avg_costs.get(ticker, 0) > 0 else 0.0
                })
                total_equity += market_val_usd
            else:
                 results.append({
                    'Ticker': ticker,
                    'Net Shares': shares,
                    'Price (Local)': 0.0,
                    'Currency': 'N/A',
                    'Market Value (USD)': 0.0
                })

        df_res = pd.DataFrame(results)
        total_portfolio = total_equity + self.cash_balance
        
        return {
            'positions': df_res,
            'total_equity': total_equity,
            'cash': self.cash_balance,
            'total_value': total_portfolio
        }

    def get_history(self, breakdown=False):
        """Calculates historical portfolio value. 
           If breakdown=True, returns DataFrame with columns for each asset."""
        # 1. Reconstruct Daily Holdings
        # Sort transactions by date
        df_txn = self.transactions.copy()
        df_txn['CreateDate'] = pd.to_datetime(df_txn['CreateDate'], errors='coerce')
        df_txn = df_txn.sort_values('CreateDate')
        
        # Get date range: First Txn to Today
        start_date = df_txn['CreateDate'].min().date()
        end_date = pd.Timestamp.now().date()
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        # Build Daily Holdings Dataframe
        # Columns = Tickers, Index = Date
        tickers = list(set(df_txn['YF_Ticker'].unique()))
        holdings_df = pd.DataFrame(0.0, index=date_range, columns=tickers)
        cash_series = pd.Series(500000.0, index=date_range)
        
        # Running state
        current_holdings = {t: 0.0 for t in tickers}
        current_cash = 500000.0
        
        # Iterate days? Or allow vectorization?
        # Vectorization is valid but logic is complex with intraday multiple txn.
        # Iterating transactions and updating the "forward" fill is better.
        
        # Initialize with 0
        # For each transaction, update holdings from that date onwards.
        # This is n_txn * n_days complexity. O(N*T). 
        # Better: Daily buckets.
        
        grouped = df_txn.groupby(df_txn['CreateDate'].dt.date)
        
        cum_holdings = pd.DataFrame(0.0, index=date_range, columns=tickers)
        cum_cash = pd.Series(0.0, index=date_range)
        
        running_h = pd.Series(0.0, index=tickers)
        running_c = 500000.0
        
        for d in date_range:
            day = d.date()
            if day in grouped.groups:
                day_txns = grouped.get_group(day)
                for _, row in day_txns.iterrows():
                    match_qty = False
                    if 'buy' in str(row['TransactionType']).lower() or 'sell' in str(row['TransactionType']).lower():
                        match_qty = True
                    
                    if match_qty:
                        ticker = row['YF_Ticker']
                        qty = row['Quantity']
                        running_h[ticker] += qty
                    
                    # Cash
                    amt = row['Amount']
                    running_c += amt
            
            cum_holdings.loc[d] = running_h
            cum_cash.loc[d] = running_c
            
        # 2. Fetch Historical Prices
        # We need prices for all tickers for the date range
        # yfinance download
        try:
            prices = yf.download(tickers, start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)['Close']
            # Forward fill missing prices (holidays/weekends)
            prices = prices.reindex(date_range).ffill().bfill()
            
            # Fill missing columns with Fallback
            for tick in tickers:
                # Force Fallback or Missing
                if tick in self.FORCE_FALLBACK or tick not in prices.columns or prices[tick].isnull().all():
                     # Use fallback
                     fb = self.fallback_prices.get(tick, self.fallback_prices.get(tick.replace('.TO','').replace('.T','').replace('.L','')))
                     if fb:
                         prices[tick] = fb['Price']
                         # Note: Fallback price is usually "LastPrice". 
                         # We assume constant for history if YF fails (e.g. Bonds).
                     else:
                         prices[tick] = 0.0
            
            # Fetch FX History
            # JPY=X, CAD=X, GBP=X
            fx_tickers = ['JPY=X', 'CAD=X', 'GBP=X']
            fx_hist = yf.download(fx_tickers, start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)['Close']
            fx_hist = fx_hist.reindex(date_range).ffill().bfill()
            
            # Handle Single Value FX result if only one? (Usually returns df with columns if list passed)
            if 'JPY=X' not in fx_hist.columns: fx_hist['JPY=X'] = 1.0 
            if 'CAD=X' not in fx_hist.columns: fx_hist['CAD=X'] = 1.0
            if 'GBP=X' not in fx_hist.columns: fx_hist['GBP=X'] = 1.0
            
        except Exception as e:
            print(f"History fetch error: {e}")
            return pd.DataFrame()
            
        # 3. Calculate Value
        total_hist_val = cum_cash.copy()
        
        for ticker in tickers:
            if ticker in prices.columns:
                # Get daily qty
                qty = cum_holdings[ticker]
                # Get daily price
                px = prices[ticker]
                
                # FX
                fx = pd.Series(1.0, index=date_range)
                if ticker.endswith('.T'):
                    fx = fx_hist['JPY=X']
                elif ticker.endswith('.TO'):
                    fx = fx_hist['CAD=X']
                elif ticker.endswith('.L'):
                    # GBPUSD=X is 1.27 (Dollars per share).
                    # GBP=X is 0.78 (Pounds per Dollar).
                    # We need to know which one YF 'GBP=X' is.
                    # Usually 'GBP=X' in Yahoo is GBP/USD rate? 
                    # Checking: JPY=X is 150 (JPY per USD).
                    # GBP=X is 0.79 (GBP per USD).
                    # So Price_USD = Price_GBP / FX.
                    fx = fx_hist['GBP=X']
                
                # Handling Pence?
                # UK stocks (.L) usually in GBp (pence).
                # If Price > 500, likely pence. If Price < 100, likely pounds?
                # HWC Price 63.
                # If 63 pence ($0.80), value is tiny.
                # If 63 Pounds ($80), value is huge.
                # Transaction Price £63.06.
                # So it's Pounds!
                # YF .L data is often in Pence. 
                # e.g. HSBA.L is 700 (pence) = 7 GBP.
                # I need to check if YF price is > 10x fallback price?
                
                # For HWC.L, let's assume it matches fallback (63).
                
                # Val = Qty * Px / FX
                val = (qty * px) / fx
                total_hist_val += val.fillna(0.0)
                
        if not breakdown:
            return total_hist_val
            
        # If breakdown requested, return DF with all columns
        # Pivot the valuations?
        # Reconstruct full history DF
        # We need Date x Ticker frame of Values (USD)
        
        history_df = pd.DataFrame(index=date_range)
        history_df['Cash'] = cum_cash
        history_df['Total'] = total_hist_val
        
        for ticker in tickers:
            if ticker in prices.columns:
                q = cum_holdings[ticker]
                p = prices[ticker]
                 # FX
                fx = pd.Series(1.0, index=date_range)
                if ticker.endswith('.T'):
                    fx = fx_hist['JPY=X']
                elif ticker.endswith('.TO'):
                    fx = fx_hist['CAD=X']
                elif ticker.endswith('.L'):
                    fx = fx_hist['GBP=X']
                
                # Handling zeros/NaNs in FX
                fx = fx.replace(0, 1.0)
                
                val = (q * p) / fx
                history_df[ticker] = val.fillna(0.0)
                
        return history_df

if __name__ == "__main__":
    # Test run
    path1 = "c:/Users/Xxran/Downloads/backtest/attachment;filename=TransactionHistory_12_13_2025.csv"
    path2 = "c:/Users/Xxran/Downloads/backtest/attachment;filename=OpenPosition_12_14_2025.csv"
    engine = PortfolioEngine(path1, path2)
    if engine.load_data():
        print(f"Cash Balance: ${engine.cash_balance:,.2f}")
        print(f"Holdings: {engine.holdings}")
        engine.fetch_market_data()
        vals = engine.get_valuations()
        print("\nPositions:")
        print(vals['positions'])
        print(f"\nTotal Portfolio Value: ${vals['total_value']:,.2f}")
    else:
        print(engine.errors)
