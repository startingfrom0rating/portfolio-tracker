
import pandas as pd
import yfinance as yf
import re
import os

class PortfolioEngine:
    # Constants
    INITIAL_CAPITAL = 500000.0
    TOTAL_COMMISSIONS = 985.00  # Total commissions paid
    INCEPTION_DATE = pd.Timestamp('2025-09-28')
    
    def __init__(self, transaction_file, open_position_file=None):
        self.transaction_file = transaction_file
        self.open_position_file = open_position_file
        self.transactions = None
        self.positions = None
        self.holdings = {} # {ticker: quantity}
        self.avg_costs = {} # {ticker: avg_cost_per_share_usd}
        self.cost_basis = {} # {ticker: total_invested_usd}
        self.cash_balance = self.INITIAL_CAPITAL
        self.market_data = {}
        self.fallback_prices = {} # {symbol: price (USD)}
        self.errors = []
        self.total_dividends = 0.0  # Track total dividends received
        self.dividend_by_ticker = {}  # {ticker: total_dividends_usd}
        self.FORCE_FALLBACK = ['HWC.L', 'HWC', 'B-T-6.250-15052030']
        self._history_cache = None  # Cache for history data
        self._sp500_cache = None  # Cache for S&P 500 data

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
        self.cash_balance = self.INITIAL_CAPITAL + total_cash_flow

        # Commissions are not represented in the CSV cash flows (user-provided total).
        # Treat them as an additional cash outflow so total value and returns are net.
        self.cash_balance -= float(self.TOTAL_COMMISSIONS)
        
        # Track dividends
        self.total_dividends = 0.0
        self.dividend_by_ticker = {}
        for _, row in self.transactions.iterrows():
            txn_type = str(row['TransactionType']).lower()
            if 'dividend' in txn_type or 'distribution' in txn_type:
                amt = row['Amount']
                ticker = row['YF_Ticker']
                self.total_dividends += amt
                self.dividend_by_ticker[ticker] = self.dividend_by_ticker.get(ticker, 0.0) + amt
        
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
        cash_series = pd.Series(self.INITIAL_CAPITAL - float(self.TOTAL_COMMISSIONS), index=date_range)
        
        # Running state
        current_holdings = {t: 0.0 for t in tickers}
        current_cash = self.INITIAL_CAPITAL - float(self.TOTAL_COMMISSIONS)
        
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
        running_c = self.INITIAL_CAPITAL - float(self.TOTAL_COMMISSIONS)
        
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
        # We need prices for all tickers for the date range.
        # Some symbols (e.g., bonds/custom IDs) are not downloadable from yfinance.
        # We'll skip those and fill using fallback prices.
        try:
            def _yf_downloadable(tk: str) -> bool:
                if tk in self.FORCE_FALLBACK:
                    return False
                # Heuristic: bond-like / internal identifiers that yfinance can't resolve.
                if tk.startswith('B-') or tk.count('-') >= 2:
                    return False
                return True

            yf_tickers = [t for t in tickers if _yf_downloadable(t)]
            if yf_tickers:
                prices = yf.download(
                    yf_tickers,
                    start=start_date,
                    end=end_date + pd.Timedelta(days=1),
                    progress=False,
                )['Close']
                if isinstance(prices, pd.Series):
                    # Single-ticker path
                    prices = prices.to_frame(name=yf_tickers[0])
            else:
                prices = pd.DataFrame(index=pd.date_range(start_date, end_date))

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

    def get_sp500_history(self, start_date=None, end_date=None):
        """Fetch benchmark history using SPY (S&P 500 ETF) as the reference."""
        if start_date is None:
            start_date = self.INCEPTION_DATE
        if end_date is None:
            end_date = pd.Timestamp.now()

        # Best-effort cache to reduce repeated yfinance calls on Streamlit reruns.
        cache = getattr(self, '_sp500_cache', None)
        if cache:
            try:
                cached_series, cached_start, cached_end = cache
                if cached_start <= pd.Timestamp(start_date).normalize() and cached_end >= pd.Timestamp(end_date).normalize():
                    sl = cached_series.loc[pd.Timestamp(start_date).normalize():pd.Timestamp(end_date).normalize()]
                    if not sl.empty:
                        return sl
            except Exception:
                pass
        
        try:
            # Use SPY as a liquid proxy for S&P 500 growth.
            raw = yf.download(
                'SPY',
                start=start_date,
                end=end_date + pd.Timedelta(days=1),
                progress=False,
            )
            
            # Handle various yfinance return formats robustly
            sp500 = pd.Series(dtype=float)
            
            if isinstance(raw, pd.DataFrame):
                # yfinance may return MultiIndex columns like ('Close', 'SPY')
                if isinstance(raw.columns, pd.MultiIndex):
                    # Flatten and find Close
                    if 'Close' in raw.columns.get_level_values(0):
                        sp500 = raw['Close'].iloc[:, 0] if isinstance(raw['Close'], pd.DataFrame) else raw['Close']
                    elif 'Adj Close' in raw.columns.get_level_values(0):
                        sp500 = raw['Adj Close'].iloc[:, 0] if isinstance(raw['Adj Close'], pd.DataFrame) else raw['Adj Close']
                else:
                    # Normal columns
                    if 'Close' in raw.columns:
                        sp500 = raw['Close']
                    elif 'Adj Close' in raw.columns:
                        sp500 = raw['Adj Close']
            
            # Ensure it's a Series
            if isinstance(sp500, pd.DataFrame):
                sp500 = sp500.iloc[:, 0]
            
            # Normalize index to midnight for alignment with our daily history index
            if not sp500.empty:
                sp500.index = pd.to_datetime(sp500.index).normalize()

            try:
                self._sp500_cache = (sp500, sp500.index.min(), sp500.index.max())
            except Exception:
                pass
            return sp500
        except Exception as e:
            print(f"Error fetching SPY benchmark: {e}")
            return pd.Series()

    def get_timeframe_returns(self):
        """Calculate returns over multiple timeframes with commission adjustment."""
        history = self.get_history(breakdown=True)
        if history.empty:
            return {}
        
        today = pd.Timestamp.now().normalize()
        
        # Find the latest available date in history
        latest_date = history.index.max()
        current_value = history['Total'].iloc[-1]
        
        # Current total value is already net of commissions because we deduct commissions from cash.
        net_current = current_value
        
        # Define timeframes
        timeframes = {
            '1D': today - pd.Timedelta(days=1),
            '1W': today - pd.Timedelta(weeks=1),
            '1M': today - pd.DateOffset(months=1),
            'YTD': pd.Timestamp(f'{today.year}-01-01'),
            'Since Sep 28': self.INCEPTION_DATE,
            'Inception': history.index.min()
        }
        
        results = {}
        for label, start_dt in timeframes.items():
            # Find closest available date
            mask = history.index >= start_dt
            if mask.any():
                start_idx = history.index[mask].min()
                start_value = history.loc[start_idx, 'Total']
                
                # For inception, use initial capital
                if label == 'Inception':
                    start_value = self.INITIAL_CAPITAL
                
                abs_return = net_current - start_value
                pct_return = ((net_current / start_value) - 1) * 100 if start_value > 0 else 0
                
                results[label] = {
                    'start_value': start_value,
                    'end_value': net_current,
                    'absolute': abs_return,
                    'percent': pct_return,
                    'start_date': start_idx
                }
        
        # Provide a dedicated net-return summary (commissions already reflected in value).
        if 'Inception' in results:
            results['Inception_Net'] = {
                'absolute': results['Inception']['absolute'],
                'percent': results['Inception']['percent'],
                'commissions': float(self.TOTAL_COMMISSIONS),
                'dividends': float(self.total_dividends),
            }
        
        return results

    def get_benchmark_comparison(self):
        """Compare portfolio performance vs SPY since Sep 28, 2025."""
        history = self.get_history(breakdown=True)
        if history.empty:
            return {}
        
        sp500 = self.get_sp500_history(start_date=self.INCEPTION_DATE)
        if sp500 is None or getattr(sp500, 'empty', True):
            return {'error': 'Could not fetch S&P 500 data'}

        # Align on normalized daily dates and since inception date
        start_dt = self.INCEPTION_DATE.normalize()
        hist_slice = history[history.index >= start_dt]
        sp_slice = sp500[sp500.index >= start_dt]

        common_dates = hist_slice.index.intersection(sp_slice.index)
        if len(common_dates) < 2:
            return {'error': 'Insufficient overlapping dates for comparison'}

        start_date = common_dates[0]
        end_date = common_dates[-1]

        port_start = float(hist_slice.loc[start_date, 'Total'])
        port_end = float(hist_slice.loc[end_date, 'Total'])
        port_return = ((port_end / port_start) - 1) * 100 if port_start else 0.0

        sp_start = float(sp_slice.loc[start_date])
        sp_end = float(sp_slice.loc[end_date])
        sp_return = ((sp_end / sp_start) - 1) * 100 if sp_start else 0.0

        excess_return = port_return - sp_return

        return {
            'portfolio_return': float(port_return),
            'sp500_return': float(sp_return),
            'excess_return': float(excess_return),
            'portfolio_start': float(port_start),
            'portfolio_end': float(port_end),
            'sp500_start': float(sp_start),
            'sp500_end': float(sp_end),
            'start_date': start_date,
            'end_date': end_date
        }

    def get_daily_attribution(self):
        """Identify which equities drove portfolio performance for the last trading day."""
        history = self.get_history(breakdown=True)
        if history.empty or len(history) < 2:
            return {'total_change': 0, 'contributors': []}
        
        # Get last two days
        today_vals = history.iloc[-1]
        yesterday_vals = history.iloc[-2]
        
        contributors = []
        total_change = today_vals['Total'] - yesterday_vals['Total']
        
        for col in history.columns:
            if col not in ['Total', 'Cash']:
                change = today_vals[col] - yesterday_vals[col]
                if abs(change) > 0.01:  # Filter noise
                    pct_contrib = (change / abs(total_change) * 100) if total_change != 0 else 0
                    contributors.append({
                        'ticker': col,
                        'change_usd': change,
                        'contribution_pct': pct_contrib,
                        'prev_value': yesterday_vals[col],
                        'curr_value': today_vals[col]
                    })
        
        # Sort by absolute contribution
        contributors.sort(key=lambda x: abs(x['change_usd']), reverse=True)
        
        return {
            'date': history.index[-1],
            'total_change': total_change,
            'total_change_pct': (total_change / yesterday_vals['Total']) * 100 if yesterday_vals['Total'] > 0 else 0,
            'contributors': contributors[:10]  # Top 10
        }

    def get_weekly_attribution(self):
        """Identify which equities drove portfolio performance for the last trading week."""
        history = self.get_history(breakdown=True)
        if history.empty:
            return {'total_change': 0, 'contributors': []}
        
        # Get data from ~7 days ago
        today = history.index.max()
        week_ago = today - pd.Timedelta(days=7)
        
        # Find closest date
        mask = history.index <= week_ago
        if not mask.any():
            week_start_idx = 0
        else:
            week_start_idx = history.index.get_loc(history.index[mask].max())
        
        today_vals = history.iloc[-1]
        week_ago_vals = history.iloc[week_start_idx]
        
        contributors = []
        total_change = today_vals['Total'] - week_ago_vals['Total']
        
        for col in history.columns:
            if col not in ['Total', 'Cash']:
                change = today_vals[col] - week_ago_vals[col]
                if abs(change) > 0.01:
                    pct_contrib = (change / abs(total_change) * 100) if total_change != 0 else 0
                    contributors.append({
                        'ticker': col,
                        'change_usd': change,
                        'contribution_pct': pct_contrib,
                        'prev_value': week_ago_vals[col],
                        'curr_value': today_vals[col]
                    })
        
        contributors.sort(key=lambda x: abs(x['change_usd']), reverse=True)
        
        return {
            'start_date': history.index[week_start_idx],
            'end_date': today,
            'total_change': total_change,
            'total_change_pct': (total_change / week_ago_vals['Total']) * 100 if week_ago_vals['Total'] > 0 else 0,
            'contributors': contributors[:10]
        }

    def get_holdings_detail(self):
        """Get detailed holdings with daily change and portfolio weight."""
        valuations = self.get_valuations()
        positions = valuations['positions'].copy()
        total_value = valuations['total_value']
        
        history = self.get_history(breakdown=True)
        
        if not positions.empty and not history.empty and len(history) >= 2:
            # Calculate daily change for each position
            today_vals = history.iloc[-1]
            yesterday_vals = history.iloc[-2]
            
            daily_changes = []
            daily_change_pcts = []
            weights = []
            
            for _, row in positions.iterrows():
                ticker = row['Ticker']
                curr_val = row['Market Value (USD)']
                
                # Portfolio weight
                weight = (curr_val / total_value * 100) if total_value > 0 else 0
                weights.append(weight)
                
                # Daily change
                if ticker in history.columns:
                    prev_val = yesterday_vals.get(ticker, curr_val)
                    change = curr_val - prev_val
                    change_pct = (change / prev_val * 100) if prev_val > 0 else 0
                else:
                    change = 0
                    change_pct = 0
                
                daily_changes.append(change)
                daily_change_pcts.append(change_pct)
            
            positions['Daily Change ($)'] = daily_changes
            positions['Daily Change (%)'] = daily_change_pcts
            positions['Weight (%)'] = weights
            
            # Add dividend info
            positions['Dividends Received'] = positions['Ticker'].map(
                lambda x: self.dividend_by_ticker.get(x, 0.0)
            )
        
        return positions

    def fetch_stock_news(self, ticker, limit=3):
        """Fetch recent news for a ticker (best effort)."""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            if news:
                return [{'title': n.get('title', ''), 'link': n.get('link', '')} for n in news[:limit]]
        except Exception:
            pass
        return []

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
