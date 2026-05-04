
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import hashlib
import tempfile
from dotenv import load_dotenv
from portfolio_engine import PortfolioEngine

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# Page Config - Minimal theme
st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="",
    layout="wide"
)

# Minimal CSS styling
st.markdown("""
<style>
    /* Clean, minimal styling */
    .stMetric {
        background: #fafafa;
        padding: 12px;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    .stMetric label {
        font-size: 12px !important;
        color: #666 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Force metric values to be visible (black text) */
    .stMetric [data-testid="stMetricValue"] {
        color: #111 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #333 !important;
    }
    h1, h2, h3 {
        font-weight: 500 !important;
    }
    .block-container {
        padding-top: 2rem;
    }
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
TRANSACTION_FILE = "attachment;filename=TransactionHistory_12_13_2025.csv"
OPEN_POSITION_FILE = "attachment;filename=OpenPosition_12_14_2025.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def get_deepseek_api_key():
    """Fetch DeepSeek API key from env or Streamlit secrets."""
    key = os.getenv("DEEPSEEK_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("DEEPSEEK_API_KEY")
    except Exception:
        return None


def as_float(x, default=0.0):
    """Coerce pandas/numpy scalars to python float."""
    try:
        if isinstance(x, pd.Series):
            if len(x) == 0:
                return float(default)
            return float(x.iloc[-1])
        return float(x)
    except Exception:
        return float(default)


def _robust_symmetric_range(values, fallback=1.0, q=0.95):
    """Compute a symmetric range for color scaling."""
    try:
        s = pd.to_numeric(values, errors='coerce').dropna()
        if s.empty:
            return float(fallback)
        hi = float(np.nanquantile(np.abs(s.values), q))
        return max(hi, float(fallback))
    except Exception:
        return float(fallback)


def get_file_signature(path):
    """Generate a lightweight signature for a local file to bust caches."""
    try:
        stat = os.stat(path)
        return f"{path}:{stat.st_size}:{int(stat.st_mtime)}"
    except Exception:
        return path or "missing"


def save_uploaded_file(uploaded_file, key_prefix):
    """Persist uploaded file to a temp path and return (path, signature, name)."""
    if uploaded_file is None:
        return None, None, None

    data = uploaded_file.getvalue()
    signature = hashlib.sha256(data).hexdigest()
    sig_key = f"{key_prefix}_signature"
    path_key = f"{key_prefix}_path"
    name_key = f"{key_prefix}_name"

    if st.session_state.get(sig_key) != signature:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(data)
        tmp.flush()
        tmp.close()
        st.session_state[sig_key] = signature
        st.session_state[path_key] = tmp.name
        st.session_state[name_key] = uploaded_file.name

    return (
        st.session_state.get(path_key),
        st.session_state.get(sig_key),
        st.session_state.get(name_key),
    )


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_equity_history(ticker, period):
    """Cached function to fetch equity price history."""
    import yfinance as yf
    try:
        if period == 'ytd':
            start_date = pd.Timestamp(f'{pd.Timestamp.now().year}-01-01')
            hist = yf.download(ticker, start=start_date, progress=False)
        else:
            hist = yf.download(ticker, period=period, progress=False)

        if hist.empty:
            return pd.DataFrame()

        # Handle multi-level columns from yf.download
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        result = hist[['Close']].copy()
        result.index = pd.to_datetime(result.index).tz_localize(None)
        return result
    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400)  # Cache for 24 hours to minimize API token usage
def get_price_movement_explanation(ticker, change_pct, news_headlines, enable_ai=True):
    """Call DeepSeek API to explain a significant price movement."""
    import requests

    if not enable_ai:
        return "AI summary disabled. Add DEEPSEEK_API_KEY to enable this insight."

    api_key = get_deepseek_api_key()
    if not api_key:
        return "AI summary disabled. Add DEEPSEEK_API_KEY to enable this insight."

    if not news_headlines:
        return "There is no clear reason for this price movement."
    
    # Format news for the prompt
    news_text = "\n".join([f"- {h}" for h in news_headlines[:5]])
    direction = "increased" if change_pct > 0 else "decreased"
    
    prompt = f"""The stock {ticker} has {direction} by {abs(change_pct):.1f}% today.

Here are recent news headlines about this company:
{news_text}

Based on these headlines, provide EXACTLY 3 concise sentences:
1. What news/event likely caused this movement
2. Brief context about the company or situation
3. Market sentiment or outlook based on the news

If none of the headlines seem relevant to explaining the price change, respond with exactly: "There is no clear reason for this price movement."

Be factual and specific. Do not use bullet points."""

    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a financial analyst assistant. Provide concise 3-sentence explanations for stock price movements based on recent news. Be factual and specific."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "max_tokens": 200
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        else:
            print(f"DeepSeek API returned status {response.status_code}: {response.text}")
            return "Unable to fetch explanation at this time."
    except Exception as e:
        print(f"DeepSeek API error: {e}")
        return "Unable to fetch explanation at this time."


@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_stock_news(ticker):
    """Fetch recent news headlines for a ticker using multiple sources with fallback."""
    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime, timedelta
    import yfinance as yf
    import time
    
    headlines = []
    fetch_source = None
    
    # Clean ticker for search (remove exchange suffixes like .T, .TO, .L)
    clean_ticker = ticker.split('.')[0] if '.' in ticker else ticker
    
    # Try Google News RSS first
    try:
        url = f"https://news.google.com/rss/search?q={clean_ticker}+stock+when:7d&hl=en-US&gl=US&ceid=US:en"
        
        for attempt in range(2):  # Retry once on failure
            response = requests.get(url, timeout=8, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')[:5]
                
                for item in items:
                    title = item.find('title')
                    pub_date = item.find('pubDate')
                    
                    if title:
                        # Check the date is within last 7 days
                        if pub_date:
                            try:
                                date_str = pub_date.text
                                # Handle multiple date formats
                                for fmt in ["%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"]:
                                    try:
                                        pub_datetime = datetime.strptime(date_str.replace('GMT', '+0000'), fmt.replace('%Z', '%z'))
                                        pub_datetime = pub_datetime.replace(tzinfo=None)
                                        if datetime.now() - pub_datetime > timedelta(days=7):
                                            continue
                                        break
                                    except:
                                        continue
                            except:
                                pass  # Include headline if date parsing fails
                        
                        headlines.append(title.text)
                
                if headlines:
                    fetch_source = "Google News"
                    break
            
            if attempt == 0:
                time.sleep(0.5)  # Brief delay before retry
    except Exception as e:
        print(f"Google News error for {ticker}: {e}")
    
    # Fallback to yfinance news if Google News returned nothing
    if not headlines:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            if news:
                for n in news[:5]:
                    # Handle both old and new yfinance news structure
                    title = None
                    if isinstance(n, dict):
                        # New structure: nested under 'content'
                        if 'content' in n and isinstance(n['content'], dict):
                            title = n['content'].get('title', '')
                        # Old structure: direct 'title' key
                        elif 'title' in n:
                            title = n.get('title', '')
                    if title:
                        headlines.append(title)
                if headlines:
                    fetch_source = "Yahoo Finance"
        except Exception as e:
            print(f"yfinance news error for {ticker}: {e}")
    
    # Store fetch source for debugging (accessible via session state)
    if 'news_fetch_sources' not in st.session_state:
        st.session_state.news_fetch_sources = {}
    st.session_state.news_fetch_sources[ticker] = fetch_source if headlines else "No articles found"
    
    return headlines


@st.cache_resource
def load_engine(transaction_path, open_position_path=None, cache_buster=None):
    """Initialize and load data into the Portfolio Engine."""
    # Force reload of the module to ensure latest code is used (e.g. new methods)
    import portfolio_engine
    import importlib
    importlib.reload(portfolio_engine)
    from portfolio_engine import PortfolioEngine
    
    eng = PortfolioEngine(transaction_path, open_position_path)
    success = eng.load_data()
    
    if not success:
        st.error(f"Failed to load data: {eng.errors}")
        return None
    
    eng.fetch_market_data()
    return eng


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_history(_engine):
    """Cached wrapper for get_history to prevent repeated yfinance calls."""
    return _engine.get_history(breakdown=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes  
def get_cached_timeframe_returns(_engine):
    """Cached wrapper for get_timeframe_returns to prevent repeated yfinance calls."""
    return _engine.get_timeframe_returns()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_benchmark_comparison(_engine, benchmark_ticker):
    """Cached wrapper for get_benchmark_comparison to prevent repeated yfinance calls."""
    return _engine.get_benchmark_comparison(benchmark_ticker)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_benchmark_history(_engine, benchmark_ticker, start_date):
    """Cached wrapper for get_benchmark_history to prevent repeated yfinance calls."""
    return _engine.get_benchmark_history(benchmark_ticker, start_date=start_date)


@st.cache_data(ttl=600)  # Cache for 10 minutes (dividends change less frequently)
def get_cached_dividend_data(_engine):
    """Cached wrapper for get_dividend_data to prevent repeated yfinance calls."""
    return _engine.get_dividend_data()


def calculate_risk_metrics(history_df, risk_free_rate=0.0):
    """Compute risk metrics and drawdown series from portfolio history."""
    if history_df is None or history_df.empty or 'Total' not in history_df.columns:
        return None

    total = history_df['Total'].dropna()
    if len(total) < 2:
        return None

    daily_returns = total.pct_change().dropna()
    if daily_returns.empty:
        return None

    days = (total.index[-1] - total.index[0]).days
    if days <= 0:
        return None

    cagr = (total.iloc[-1] / total.iloc[0]) ** (365 / days) - 1
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / vol if vol and vol > 0 else np.nan

    running_max = total.cummax()
    drawdown = (total / running_max) - 1.0
    max_drawdown = drawdown.min()

    return {
        'cagr': cagr,
        'volatility': vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'drawdown_series': drawdown,
        'best_day': daily_returns.max(),
        'worst_day': daily_returns.min(),
    }


def calculate_beta_and_corr(portfolio_series, benchmark_series):
    """Calculate beta and correlation between portfolio and benchmark returns."""
    if portfolio_series is None or benchmark_series is None:
        return None, None

    aligned = pd.concat(
        [portfolio_series.rename('portfolio'), benchmark_series.rename('benchmark')],
        axis=1
    ).dropna()

    if len(aligned) < 3:
        return None, None

    returns = aligned.pct_change().dropna()
    if returns.empty:
        return None, None

    cov = returns['portfolio'].cov(returns['benchmark'])
    var = returns['benchmark'].var()
    beta = cov / var if var and var > 0 else None
    corr = returns['portfolio'].corr(returns['benchmark'])

    return beta, corr


def calculate_contribution_data(history_df, start_date):
    """Compute contribution by asset between start_date and latest."""
    if history_df is None or history_df.empty:
        return pd.DataFrame(), None, None

    end_date = history_df.index.max()
    mask = history_df.index >= start_date
    if mask.any():
        start_idx = history_df.index[mask].min()
    else:
        start_idx = history_df.index.min()

    start_vals = history_df.loc[start_idx]
    end_vals = history_df.loc[end_date]

    total_change = end_vals.get('Total', 0) - start_vals.get('Total', 0)
    delta = (end_vals - start_vals).drop(labels=['Total'], errors='ignore')

    contrib_df = delta.reset_index()
    contrib_df.columns = ['Ticker', 'Change']
    if total_change != 0:
        contrib_df['Contribution %'] = contrib_df['Change'] / total_change * 100
    else:
        contrib_df['Contribution %'] = 0.0

    contrib_df = contrib_df.sort_values('Change', ascending=False)

    return contrib_df, start_idx, end_date


def render_overview_tab(engine, valuation_data, history_df, timeframe_returns):
    """Render the Overview tab."""
    total_val = valuation_data['total_value']
    positions_df = valuation_data['positions']
    
    # Key Metrics
    st.subheader("Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric(
        "Portfolio Value",
        f"${total_val:,.2f}",
        f"{timeframe_returns.get('1D', {}).get('percent', 0):+.2f}% (1D)"
    )
    
    inception_net = timeframe_returns.get('Inception_Net', {})
    net_return = inception_net.get('absolute', 0)
    net_return_pct = inception_net.get('percent', 0)
    col2.metric(
        "Net Return",
        f"${net_return:,.2f}",
        f"{net_return_pct:+.2f}%"
    )
    
    col3.metric("Total Dividends", f"${engine.total_dividends:,.2f}")
    col4.metric("Total Commissions", f"${engine.TOTAL_COMMISSIONS:,.2f}")
    col5.metric("Cash Balance", f"${valuation_data['cash']:,.2f}")
    
    st.markdown("---")
    
    # Returns Table
    st.subheader("Performance Returns")
    
    returns_data = []
    for tf in ['1D', '1W', '1M', 'YTD', 'Since Sep 28', 'Inception']:
        if tf in timeframe_returns:
            r = timeframe_returns[tf]
            returns_data.append({
                'Timeframe': tf,
                'Return ($)': r.get('absolute', 0),
                'Return (%)': r.get('percent', 0)
            })
    
    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        
        def color_returns(val):
            if isinstance(val, (int, float)):
                return 'color: #2e7d32' if val >= 0 else 'color: #c62828'
            return ''
        
        styled_returns = returns_df.style.format({
            'Return ($)': '${:+,.2f}',
            'Return (%)': '{:+.2f}%'
        }).map(color_returns, subset=['Return ($)', 'Return (%)'])
        
        st.dataframe(styled_returns, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Portfolio Growth Chart
    st.subheader("Portfolio Growth")
    if not history_df.empty:
        fig_hist = px.line(
            history_df, 
            x=history_df.index, 
            y='Total',
            labels={'Total': 'Value ($)', 'index': 'Date'}
        )
        fig_hist.update_layout(
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=300,
            xaxis_title='',
            yaxis_title='Value (USD)',
            yaxis_tickprefix='$'
        )
        fig_hist.update_traces(line_color='#1976d2')
        st.plotly_chart(fig_hist, use_container_width=True)

        export_df = history_df.reset_index().rename(columns={'index': 'Date'})
        csv_history = export_df.to_csv(index=False)
        st.download_button(
            "Download portfolio history (CSV)",
            csv_history,
            file_name="portfolio_history.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # Two column layout for charts
    col_alloc, col_pnl = st.columns(2)
    
    with col_alloc:
        st.subheader("Allocation")
        if not positions_df.empty:
            df_hm = positions_df.copy()
            df_hm['PnL_Display'] = df_hm['Unrealized PnL'].apply(
                lambda x: f"+${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"
            )
            df_hm['Pct_Display'] = df_hm['PnL %'].apply(lambda x: f"{x:+.1f}%")
            
            fig_tree = px.treemap(
                df_hm,
                path=[px.Constant("Portfolio"), 'Ticker'],
                values='Market Value (USD)',
                color='PnL %',
                color_continuous_scale=[
                    [0.0, '#c62828'],
                    [0.4, '#ef9a9a'],
                    [0.5, '#f5f5f5'],
                    [0.6, '#a5d6a7'],
                    [1.0, '#2e7d32']
                ],
                color_continuous_midpoint=0,
            )
            fig_tree.update_traces(
                textinfo="label+value+percent parent",
                textfont=dict(size=11, color='#333'),
                marker=dict(line=dict(width=1, color='white')),
            )
            fig_tree.update_layout(
                height=400,
                margin=dict(t=10, l=10, r=10, b=10),
                coloraxis_colorbar=dict(title="PnL %", ticksuffix="%", len=0.6)
            )
            st.plotly_chart(fig_tree, use_container_width=True)
    
    with col_pnl:
        st.subheader("Unrealized PnL by Asset")
        if not positions_df.empty:
            fig_pnl = px.bar(
                positions_df.sort_values('Unrealized PnL'),
                x='Ticker',
                y='Unrealized PnL',
                color='Unrealized PnL',
                color_continuous_scale='RdBu',
                text_auto='.2s'
            )
            fig_pnl.update_traces(textposition='outside')
            fig_pnl.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                xaxis_title='',
                yaxis_title='PnL (USD)',
                yaxis_tickprefix='$'
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

    st.markdown("---")

    st.subheader("Portfolio Breakdown")
    col_currency, col_concentration = st.columns(2)

    with col_currency:
        st.markdown("**Currency Exposure**")
        currency_df = engine.get_currency_exposure(include_cash=True)
        if not currency_df.empty:
            fig_currency = px.pie(
                currency_df,
                names='Currency',
                values='Market Value (USD)',
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            fig_currency.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=True
            )
            st.plotly_chart(fig_currency, use_container_width=True)
            st.dataframe(
                currency_df[['Currency', 'Market Value (USD)', 'Weight (%)']].style.format({
                    'Market Value (USD)': '${:,.2f}',
                    'Weight (%)': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.caption("Currency exposure unavailable.")

    with col_concentration:
        st.markdown("**Concentration Risk**")
        concentration = engine.get_concentration_metrics(top_n=5)
        if concentration:
            top = concentration['top_holdings']
            col1, col2, col3 = st.columns(3)
            col1.metric("Top 5 Weight", f"{concentration['top_weight']:.2f}%")
            col2.metric("HHI", f"{concentration['hhi']:.3f}")
            eff_n = concentration.get('effective_n')
            col3.metric("Effective N", f"{eff_n:.1f}" if eff_n else "N/A")

            fig_top = px.bar(
                top.sort_values('Weight (%)'),
                x='Weight (%)',
                y='Ticker',
                orientation='h',
                text=top['Weight (%)'].map(lambda x: f"{x:.1f}%"),
                color='Weight (%)',
                color_continuous_scale='Blues'
            )
            fig_top.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title='Weight (%)',
                yaxis_title='',
                showlegend=False
            )
            fig_top.update_traces(textposition='outside')
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.caption("Concentration metrics unavailable.")


def render_holdings_tab(engine, history_df):
    """Render the Holdings tab."""
    st.subheader("Current Holdings")
    
    holdings_df = engine.get_holdings_detail()
    
    if holdings_df.empty:
        st.info("No active positions.")
        return
    
    holdings_df = holdings_df.sort_values('Market Value (USD)', ascending=False)
    
    # Check if any tickers are using stale fallback data
    stale_tickers = getattr(engine, 'stale_tickers', [])
    if stale_tickers:
        st.warning(f"⚠️ **Rate limited by Yahoo Finance.** The following tickers are using cached prices from Dec 14, 2025: {', '.join(stale_tickers)}. Prices and daily changes may be outdated.")
    
    # Store significant movers data for the Summary tab (don't render inline)
    if 'Daily Change (%)' in holdings_df.columns:
        # Filter for valid significant movers:
        # - Daily change >= 3% or <= -3%
        # - Exclude 0% (no data / rate limited)
        # - Exclude extreme values like -100% (data errors)
        # - Exclude very small positions (< $100 value)
        # - Exclude tickers using stale fallback data
        valid_mask = (
            (holdings_df['Daily Change (%)'].abs() >= 3.0) & 
            (holdings_df['Daily Change (%)'].abs() < 50.0) &  # Filter out data errors
            (holdings_df['Daily Change (%)'] != 0.0) &  # Filter out no-data
            (holdings_df['Market Value (USD)'] >= 100.0) &  # Filter tiny positions
            (~holdings_df['Ticker'].isin(stale_tickers))  # Exclude stale data tickers
        )
        significant_movers = holdings_df[valid_mask]
        
        # Store in session state for the Summary tab
        st.session_state.significant_movers = significant_movers
        st.session_state.stale_tickers = stale_tickers
        
        # Show a brief indicator if there are significant movers
        if not significant_movers.empty:
            mover_count = len(significant_movers)
            st.info(f"📊 **{mover_count} significant mover(s) today** — View AI analysis in the **Summary** tab")
    
    display_cols = [
        'Ticker', 'Net Shares', 'Price (Local)', 'Currency',
        'Market Value (USD)', 'Avg Cost', 'Unrealized PnL', 'PnL %',
        'Daily Change ($)', 'Daily Change (%)', 'Weight (%)', 'Dividends Received'
    ]
    
    display_cols = [c for c in display_cols if c in holdings_df.columns]
    display_df = holdings_df[display_cols].copy()
    
    format_dict = {
        "Net Shares": "{:,.2f}",
        "Price (Local)": "{:,.2f}",
        "Market Value (USD)": "${:,.2f}",
        "Avg Cost": "${:,.2f}",
        "Unrealized PnL": "${:+,.2f}",
        "PnL %": "{:+.2f}%",
        "Daily Change ($)": "${:+,.2f}",
        "Daily Change (%)": "{:+.2f}%",
        "Weight (%)": "{:.1f}%",
        "Dividends Received": "${:,.2f}"
    }
    
    active_format = {k: v for k, v in format_dict.items() if k in display_df.columns}
    styler = display_df.style.format(active_format)
    
    try:
        import matplotlib
        if 'Unrealized PnL' in display_df.columns:
            v = _robust_symmetric_range(display_df['Unrealized PnL'], fallback=5000.0, q=0.95)
            styler = styler.background_gradient(subset=['Unrealized PnL'], cmap="RdBu", vmin=-v, vmax=v)
        if 'Daily Change ($)' in display_df.columns:
            v = _robust_symmetric_range(display_df['Daily Change ($)'], fallback=500.0, q=0.95)
            styler = styler.background_gradient(subset=['Daily Change ($)'], cmap="RdBu", vmin=-v, vmax=v)
        if 'Weight (%)' in display_df.columns:
            styler = styler.bar(subset=['Weight (%)'], color='#90caf9', vmin=0)
    except Exception:
        pass
    
    st.dataframe(styler, use_container_width=True, height=500, hide_index=True)

    csv_holdings = holdings_df.to_csv(index=False)
    st.download_button(
        "Download holdings (CSV)",
        csv_holdings,
        file_name="holdings.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Position Details
    st.subheader("Position Details")
    
    tickers = holdings_df['Ticker'].tolist()
    selected = st.selectbox("Select Position", tickers)
    
    if selected:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Timespan selection buttons
            timespan_col1, timespan_col2, timespan_col3, timespan_col4 = st.columns(4)
            
            with timespan_col1:
                btn_1m = st.button("1M", key="btn_1m", use_container_width=True)
            with timespan_col2:
                btn_6m = st.button("6M", key="btn_6m", use_container_width=True)
            with timespan_col3:
                btn_1y = st.button("1Y", key="btn_1y", use_container_width=True)
            with timespan_col4:
                btn_ytd = st.button("YTD", key="btn_ytd", use_container_width=True)
            
            # Determine selected period
            if 'position_timespan' not in st.session_state:
                st.session_state.position_timespan = '1y'
            
            if btn_1m:
                st.session_state.position_timespan = '1mo'
            elif btn_6m:
                st.session_state.position_timespan = '6mo'
            elif btn_1y:
                st.session_state.position_timespan = '1y'
            elif btn_ytd:
                st.session_state.position_timespan = 'ytd'
            
            period = st.session_state.position_timespan
            
            # Fetch equity price history (using cached function)
            with st.spinner(f"Loading {selected} price data..."):
                price_history = fetch_equity_history(selected, period)
            
            if not price_history.empty:
                # Get buy transactions to mark on chart
                buy_transactions = engine.get_buy_transactions(selected)
                
                # Get currency for the ticker
                pos_data = holdings_df[holdings_df['Ticker'] == selected].iloc[0]
                currency = pos_data.get('Currency', 'USD')
                currency_symbol = {'USD': '$', 'CAD': 'C$', 'JPY': '¥', 'GBP': '£'}.get(currency, '$')
                
                # Create the figure
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(go.Scatter(
                    x=price_history.index,
                    y=price_history['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#1976d2', width=2),
                    hovertemplate=f'{currency_symbol}%{{y:.2f}}<extra></extra>'
                ))
                
                # Add buy markers (green triangles)
                if buy_transactions:
                    chart_start = price_history.index.min()
                    chart_end = price_history.index.max()
                    
                    for buy in buy_transactions:
                        buy_date = buy['date']
                        # Only show buys within the chart's date range
                        if chart_start <= buy_date <= chart_end:
                            buy_price = buy['price']
                            buy_qty = buy['quantity']
                            
                            fig.add_trace(go.Scatter(
                                x=[buy_date],
                                y=[buy_price],
                                mode='markers',
                                name=f"Buy ({buy_date.strftime('%Y-%m-%d')})",
                                marker=dict(
                                    symbol='triangle-up',
                                    size=14,
                                    color='#2e7d32',
                                    line=dict(width=1, color='white')
                                ),
                                hovertemplate=(
                                    f"<b>BUY</b><br>"
                                    f"Date: {buy_date.strftime('%Y-%m-%d')}<br>"
                                    f"Price: {currency_symbol}{buy_price:.2f}<br>"
                                    f"Qty: {buy_qty:,.0f}<extra></extra>"
                                ),
                                showlegend=False
                            ))
                
                # Period label for title
                period_labels = {'1mo': '1 Month', '6mo': '6 Months', '1y': '1 Year', 'ytd': 'Year to Date'}
                period_label = period_labels.get(period, period)
                
                fig.update_layout(
                    title=f'{selected} - Price History ({period_label})',
                    height=320,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis_title='',
                    yaxis_title=f'Price ({currency})',
                    yaxis_tickprefix=currency_symbol,
                    hovermode='x unified',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Price history unavailable for {selected}")
        
        with col2:
            pos_data = holdings_df[holdings_df['Ticker'] == selected].iloc[0]
            st.markdown(f"**{selected}**")
            st.write(f"Shares: {pos_data['Net Shares']:,.2f}")
            st.write(f"Market Value: ${pos_data['Market Value (USD)']:,.2f}")
            st.write(f"Avg Cost: ${pos_data['Avg Cost']:,.2f}")
            st.write(f"P&L: ${pos_data['Unrealized PnL']:+,.2f} ({pos_data['PnL %']:+.2f}%)")
            if 'Weight (%)' in pos_data:
                st.write(f"Weight: {pos_data['Weight (%)']:.1f}%")
            if 'Dividends Received' in pos_data:
                st.write(f"Dividends: ${pos_data['Dividends Received']:,.2f}")
            
            # Show buy history summary
            buy_transactions = engine.get_buy_transactions(selected)
            if buy_transactions:
                st.markdown("---")
                st.markdown("**Purchase History**")
                for buy in buy_transactions[-5:]:  # Show last 5 buys
                    currency = buy['currency']
                    currency_symbol = {'USD': '$', 'CAD': 'C$', 'JPY': '¥', 'GBP': '£'}.get(currency, '$')
                    st.caption(f"{buy['date'].strftime('%Y-%m-%d')}: {buy['quantity']:,.0f} @ {currency_symbol}{buy['price']:.2f}")


def render_transactions_tab(engine):
    """Render the Transactions tab with filters and summaries."""
    st.subheader("Transactions")

    if engine.transactions is None or engine.transactions.empty:
        st.info("No transactions available.")
        return

    df = engine.transactions.copy()
    df = df.sort_values('CreateDate', ascending=False)

    min_date = df['CreateDate'].min().date() if pd.notna(df['CreateDate']).any() else pd.Timestamp.now().date()
    max_date = df['CreateDate'].max().date() if pd.notna(df['CreateDate']).any() else pd.Timestamp.now().date()

    col1, col2, col3 = st.columns(3)
    with col1:
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        tickers = sorted(df['YF_Ticker'].dropna().unique().tolist())
        ticker_filter = st.multiselect("Ticker", tickers, default=[])
    with col3:
        types = sorted(df['TransactionType'].dropna().unique().tolist())
        type_filter = st.multiselect("Transaction type", types, default=[])

    search_term = st.text_input("Search (symbol / type)")

    filtered = df.copy()

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered['CreateDate'].dt.date >= start_date) &
            (filtered['CreateDate'].dt.date <= end_date)
        ]

    if ticker_filter:
        filtered = filtered[filtered['YF_Ticker'].isin(ticker_filter)]

    if type_filter:
        filtered = filtered[filtered['TransactionType'].isin(type_filter)]

    if search_term:
        search_lower = search_term.lower()
        filtered = filtered[
            filtered['Symbol'].astype(str).str.lower().str.contains(search_lower) |
            filtered['TransactionType'].astype(str).str.lower().str.contains(search_lower)
        ]

    buys = filtered[filtered['TransactionType'].str.lower().str.contains('buy', na=False)]
    sells = filtered[filtered['TransactionType'].str.lower().str.contains('sell', na=False)]
    divs = filtered[filtered['TransactionType'].str.lower().str.contains('dividend|distribution', na=False)]

    total_buys = -buys['Amount'].sum() if not buys.empty else 0.0
    total_sells = sells['Amount'].sum() if not sells.empty else 0.0
    total_divs = divs['Amount'].sum() if not divs.empty else 0.0
    net_cash = filtered['Amount'].sum() if not filtered.empty else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Buys", f"${total_buys:,.2f}")
    m2.metric("Total Sells", f"${total_sells:,.2f}")
    m3.metric("Dividends", f"${total_divs:,.2f}")
    m4.metric("Net Cash Flow", f"${net_cash:,.2f}")

    st.caption(f"{len(filtered):,} transactions shown")

    display_cols = [
        'CreateDate', 'YF_Ticker', 'Symbol', 'TransactionType',
        'Quantity', 'Price', 'Amount', 'Currency', 'FXRate'
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    styled = filtered[display_cols].copy()
    if 'CreateDate' in styled.columns:
        styled['CreateDate'] = styled['CreateDate'].dt.strftime('%Y-%m-%d')

    st.dataframe(
        styled.style.format({
            'Quantity': '{:,.2f}',
            'Price': '{:,.4f}',
            'Amount': '${:+,.2f}',
            'FXRate': '{:,.4f}'
        }),
        use_container_width=True,
        hide_index=True,
        height=500
    )

    csv_txn = filtered.to_csv(index=False)
    st.download_button(
        "Download transactions (CSV)",
        csv_txn,
        file_name="transactions.csv",
        mime="text/csv"
    )


def render_analysis_tab(engine, history_df, risk_free_rate, benchmark_ticker):
    """Render the Analysis tab."""
    
    # Benchmark Comparison
    st.subheader("Performance vs Benchmark")

    benchmark_ticker = (benchmark_ticker or 'SPY').upper().strip()
    timeframe = st.selectbox(
        "Timeframe",
        ["1M", "3M", "6M", "1Y", "YTD", "Inception"],
        index=5,
        key="benchmark_timeframe"
    )

    today = pd.Timestamp.now().normalize()
    if timeframe == "1M":
        start_dt = today - pd.DateOffset(months=1)
    elif timeframe == "3M":
        start_dt = today - pd.DateOffset(months=3)
    elif timeframe == "6M":
        start_dt = today - pd.DateOffset(months=6)
    elif timeframe == "1Y":
        start_dt = today - pd.DateOffset(years=1)
    elif timeframe == "YTD":
        start_dt = pd.Timestamp(f"{today.year}-01-01")
    else:
        start_dt = engine.INCEPTION_DATE.normalize() if hasattr(engine, "INCEPTION_DATE") else history_df.index.min()

    hist_slice = history_df[history_df.index >= start_dt] if history_df is not None else pd.DataFrame()
    benchmark_hist = get_cached_benchmark_history(engine, benchmark_ticker, start_dt)

    chart_rendered = False
    if benchmark_hist is not None and len(benchmark_hist) > 0 and not hist_slice.empty:
        if isinstance(benchmark_hist, pd.DataFrame):
            benchmark_hist = benchmark_hist.iloc[:, 0]

        common_dates = hist_slice.index.intersection(benchmark_hist.index)

        if len(common_dates) > 1:
            base_date = common_dates[0]
            port_base = as_float(hist_slice.loc[base_date, 'Total'], default=1.0)
            bench_base = as_float(benchmark_hist.loc[base_date], default=1.0)

            if port_base > 0 and bench_base > 0:
                port_ret_series = ((hist_slice.loc[common_dates, 'Total'] / port_base) - 1.0) * 100
                bench_ret_series = ((benchmark_hist.loc[common_dates] / bench_base) - 1.0) * 100

                port_ret = float(port_ret_series.iloc[-1])
                bench_ret = float(bench_ret_series.iloc[-1])
                excess = port_ret - bench_ret

                col1, col2, col3 = st.columns(3)
                col1.metric("Portfolio", f"{port_ret:+.2f}%", delta=timeframe, delta_color="off")
                col2.metric(f"{benchmark_ticker}", f"{bench_ret:+.2f}%", delta=timeframe, delta_color="off")
                col3.metric(
                    "Excess Return",
                    f"{excess:+.2f}%",
                    delta="Outperforming" if excess >= 0 else "Underperforming",
                    delta_color="normal" if excess >= 0 else "inverse"
                )

                beta, corr = calculate_beta_and_corr(
                    hist_slice.loc[common_dates, 'Total'],
                    benchmark_hist.loc[common_dates]
                )
                col_beta, col_corr = st.columns(2)
                col_beta.metric("Beta", f"{beta:.2f}" if beta is not None else "N/A")
                col_corr.metric("Correlation", f"{corr:.2f}" if corr is not None else "N/A")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=port_ret_series.index,
                    y=port_ret_series.values,
                    name='Portfolio',
                    line=dict(color='#1976d2', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(25, 118, 210, 0.08)',
                ))

                fig.add_trace(go.Scatter(
                    x=bench_ret_series.index,
                    y=bench_ret_series.values,
                    name=benchmark_ticker,
                    line=dict(color='#757575', width=2, dash='dot'),
                ))

                fig.add_hline(y=0, line_dash="dash", line_color="#bdbdbd", opacity=0.5)

                fig.update_layout(
                    title=f'Cumulative Return: Portfolio vs {benchmark_ticker}',
                    xaxis_title='',
                    yaxis_title='Return (%)',
                    yaxis=dict(ticksuffix='%'),
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)
                chart_rendered = True

    if not chart_rendered:
        st.info("Benchmark chart unavailable - insufficient data.")
    
    st.markdown("---")

    # Risk & Drawdown
    st.subheader("Risk & Drawdown")
    risk_metrics = calculate_risk_metrics(history_df, risk_free_rate=risk_free_rate)

    if risk_metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CAGR", f"{risk_metrics['cagr'] * 100:.2f}%")
        col2.metric("Volatility", f"{risk_metrics['volatility'] * 100:.2f}%")

        sharpe_val = risk_metrics['sharpe']
        sharpe_text = f"{sharpe_val:.2f}" if np.isfinite(sharpe_val) else "N/A"
        col3.metric("Sharpe", sharpe_text)
        col4.metric("Max Drawdown", f"{risk_metrics['max_drawdown'] * 100:.2f}%")

        dd = risk_metrics['drawdown_series'] * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values,
            name='Drawdown',
            line=dict(color='#c62828', width=2),
            fill='tozeroy',
            fillcolor='rgba(198, 40, 40, 0.15)'
        ))
        fig_dd.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title='Drawdown (%)',
            xaxis_title='',
            yaxis=dict(ticksuffix='%'),
            hovermode='x unified'
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        st.caption(
            f"Best day: {risk_metrics['best_day'] * 100:+.2f}% | "
            f"Worst day: {risk_metrics['worst_day'] * 100:+.2f}%"
        )
    else:
        st.info("Risk metrics unavailable - insufficient history.")

    st.markdown("---")
    
    # Daily Performance
    st.subheader("Today's Performance")
    
    daily = engine.get_daily_attribution()
    
    if abs(daily.get('total_change', 0)) > 0.01:
        total_chg = daily['total_change']
        total_pct = daily['total_change_pct']
        is_positive = total_chg >= 0
        
        chg_color = "#2e7d32" if is_positive else "#c62828"
        sign = "+" if is_positive else ""
        
        st.markdown(f"""
        <div style="background: #fafafa; padding: 16px; border-radius: 4px; border-left: 3px solid {chg_color}; margin-bottom: 16px;">
            <span style="font-size: 20px; font-weight: 500; color: {chg_color};">
                {sign}${abs(total_chg):,.2f} ({total_pct:+.2f}%)
            </span>
            <span style="color: #666; font-size: 13px; margin-left: 12px;">Day over Day</span>
        </div>
        """, unsafe_allow_html=True)
        
        contributors = daily.get('contributors', [])
        if contributors:
            winners = sorted([c for c in contributors if c['change_usd'] > 0], key=lambda x: x['change_usd'], reverse=True)[:5]
            losers = sorted([c for c in contributors if c['change_usd'] < 0], key=lambda x: x['change_usd'])[:5]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Gainers**")
                if winners:
                    for w in winners:
                        st.markdown(f"<div style='display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #eee;'><span>{w['ticker']}</span><span style='color: #2e7d32;'>+${w['change_usd']:,.0f}</span></div>", unsafe_allow_html=True)
                else:
                    st.caption("No gainers today")
            
            with col2:
                st.markdown("**Top Losers**")
                if losers:
                    for l in losers:
                        st.markdown(f"<div style='display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #eee;'><span>{l['ticker']}</span><span style='color: #c62828;'>${l['change_usd']:,.0f}</span></div>", unsafe_allow_html=True)
                else:
                    st.caption("No losers today")
    else:
        st.info("Market closed or no significant movement today.")
    
    st.markdown("---")
    
    # Weekly Performance
    st.subheader("Weekly Performance")
    
    weekly = engine.get_weekly_attribution()
    
    if abs(weekly.get('total_change', 0)) > 0.01:
        total_chg = weekly['total_change']
        total_pct = weekly['total_change_pct']
        is_positive = total_chg >= 0
        
        chg_color = "#2e7d32" if is_positive else "#c62828"
        sign = "+" if is_positive else ""
        
        st.markdown(f"""
        <div style="background: #fafafa; padding: 16px; border-radius: 4px; border-left: 3px solid {chg_color}; margin-bottom: 16px;">
            <span style="font-size: 20px; font-weight: 500; color: {chg_color};">
                {sign}${abs(total_chg):,.2f} ({total_pct:+.2f}%)
            </span>
            <span style="color: #666; font-size: 13px; margin-left: 12px;">Week over Week</span>
        </div>
        """, unsafe_allow_html=True)
        
        contributors = weekly.get('contributors', [])
        if contributors:
            contrib_df = pd.DataFrame(contributors)
            contrib_df = contrib_df.sort_values('change_usd', key=lambda x: x.abs(), ascending=True).tail(10)
            
            colors = ['#2e7d32' if x >= 0 else '#c62828' for x in contrib_df['change_usd']]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=contrib_df['ticker'],
                x=contrib_df['change_usd'],
                orientation='h',
                marker_color=colors,
                text=[f"${x:+,.0f}" for x in contrib_df['change_usd']],
                textposition='outside',
            ))
            
            fig.update_layout(
                title='Weekly Attribution',
                xaxis_title='Impact ($)',
                yaxis_title='',
                height=max(250, len(contrib_df) * 30),
                margin=dict(l=0, r=60, t=40, b=40),
                showlegend=False
            )
            
            fig.add_vline(x=0, line_color="#bdbdbd", line_dash="dash", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No significant portfolio movement this week.")
    
    st.markdown("---")

    # Contribution Analysis
    st.subheader("Contribution by Asset")
    contrib_period = st.selectbox(
        "Contribution period",
        ["1W", "1M", "3M", "6M", "1Y", "YTD", "Inception"],
        index=2,
        key="contrib_period"
    )

    if contrib_period == "1W":
        contrib_start = today - pd.Timedelta(days=7)
    elif contrib_period == "1M":
        contrib_start = today - pd.DateOffset(months=1)
    elif contrib_period == "3M":
        contrib_start = today - pd.DateOffset(months=3)
    elif contrib_period == "6M":
        contrib_start = today - pd.DateOffset(months=6)
    elif contrib_period == "1Y":
        contrib_start = today - pd.DateOffset(years=1)
    elif contrib_period == "YTD":
        contrib_start = pd.Timestamp(f"{today.year}-01-01")
    else:
        contrib_start = engine.INCEPTION_DATE.normalize() if hasattr(engine, "INCEPTION_DATE") else history_df.index.min()

    contrib_df, contrib_start_idx, contrib_end = calculate_contribution_data(history_df, contrib_start)

    if not contrib_df.empty:
        top = contrib_df.reindex(
            contrib_df['Change'].abs().sort_values(ascending=False).head(12).index
        )

        colors = ['#2e7d32' if x >= 0 else '#c62828' for x in top['Change']]
        fig_contrib = go.Figure()
        fig_contrib.add_trace(go.Bar(
            x=top['Change'],
            y=top['Ticker'],
            orientation='h',
            marker_color=colors,
            text=[f"${x:+,.0f}" for x in top['Change']],
            textposition='outside'
        ))
        fig_contrib.update_layout(
            height=max(320, len(top) * 28),
            margin=dict(l=0, r=60, t=20, b=10),
            xaxis_title='Change (USD)',
            yaxis_title='',
            showlegend=False
        )
        fig_contrib.add_vline(x=0, line_color="#bdbdbd", line_dash="dash", opacity=0.5)
        st.plotly_chart(fig_contrib, use_container_width=True)

        st.caption(f"From {contrib_start_idx.strftime('%Y-%m-%d')} to {contrib_end.strftime('%Y-%m-%d')}")

        contrib_display = contrib_df.copy()
        st.dataframe(
            contrib_display.style.format({
                'Change': '${:+,.2f}',
                'Contribution %': '{:+.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

        csv_contrib = contrib_df.to_csv(index=False)
        st.download_button(
            "Download contribution data (CSV)",
            csv_contrib,
            file_name="contribution_analysis.csv",
            mime="text/csv"
        )
    else:
        st.info("Contribution analysis unavailable - insufficient history.")

    st.markdown("---")
    
    # Asset Drill-Down
    st.subheader("Asset Performance")
    
    if not history_df.empty:
        tickers = ['Total', 'Cash'] + sorted([c for c in history_df.columns if c not in ['Total', 'Cash']])
        sel_ticker = st.selectbox("Select asset", tickers, key="analysis_drilldown")
        
        if sel_ticker:
            fig = px.area(
                history_df,
                x=history_df.index,
                y=sel_ticker,
                color_discrete_sequence=['#1976d2']
            )
            fig.update_layout(
                xaxis_title='',
                yaxis_title='Value (USD)',
                yaxis_tickprefix='$',
                height=280,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)


def render_summary_tab():
    """Render the AI Summary tab with significant movers analysis."""
    st.subheader("📊 AI Market Summary")
    st.caption("DeepSeek-powered analysis of significant price movements (≥3% daily change)")

    api_key_present = bool(get_deepseek_api_key())
    if not api_key_present:
        st.warning("AI summary is disabled. Add DEEPSEEK_API_KEY to .env or Streamlit secrets to enable.")
    
    # Get significant movers from session state (populated by holdings tab)
    significant_movers = st.session_state.get('significant_movers', pd.DataFrame())
    
    if significant_movers is None or (isinstance(significant_movers, pd.DataFrame) and significant_movers.empty):
        st.info("No significant movers today (positions with ≥3% daily change and ≥$100 value).")
        return
    
    st.markdown("---")
    
    # Process each significant mover
    for idx, (_, row) in enumerate(significant_movers.iterrows()):
        ticker = row['Ticker']
        change_pct = row['Daily Change (%)']
        change_usd = row.get('Daily Change ($)', 0)
        market_value = row.get('Market Value (USD)', 0)
        
        # Fetch news and get AI explanation
        news = fetch_stock_news(ticker)
        explanation = get_price_movement_explanation(ticker, change_pct, news, enable_ai=api_key_present)
        
        # Get news source info for debugging
        news_sources = st.session_state.get('news_fetch_sources', {})
        news_source = news_sources.get(ticker, "Unknown")
        
        # Determine styling
        if change_pct >= 0:
            bg_color = "#d4edda"
            border_color = "#1acc44"
            icon = "📈"
            direction = "up"
        else:
            bg_color = "#f8d7da"
            border_color = "#f7142b"
            icon = "📉"
            direction = "down"
        
        headline_block = ""
        if not api_key_present and news:
            headlines_html = "".join([f"<li>{h}</li>" for h in news[:3]])
            headline_block = (
                "<div style='margin-top: 8px; color: #444; font-size: 13px;'>"
                "<strong>Recent headlines</strong>"
                f"<ul style='margin: 6px 0 0 18px;'>{headlines_html}</ul>"
                "</div>"
            )

        # Render the analysis card
        st.markdown(f"""
        <div style="background: {bg_color}; padding: 16px 20px; border-radius: 8px; border-left: 5px solid {border_color}; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 18px; font-weight: 600;">{icon} {ticker}</span>
                <span style="color: {border_color}; font-weight: 700; font-size: 16px;">{change_pct:+.2f}%</span>
            </div>
            <div style="color: #555; font-size: 13px; margin-bottom: 10px;">
                Daily P&L: <strong>${change_usd:+,.2f}</strong> &nbsp;|&nbsp; Position Value: <strong>${market_value:,.2f}</strong>
            </div>
            <div style="color: #333; font-size: 14px; line-height: 1.5;">
                {explanation}
            </div>
            {headline_block}
            <div style="color: #888; font-size: 11px; margin-top: 8px;">
                News source: {news_source} ({len(news)} article(s) found)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show summary statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    gainers = significant_movers[significant_movers['Daily Change (%)'] > 0]
    losers = significant_movers[significant_movers['Daily Change (%)'] < 0]
    
    with col1:
        st.metric("Total Significant Movers", len(significant_movers))
    with col2:
        st.metric("Gainers (≥3%)", len(gainers), delta=f"+{len(gainers)}" if len(gainers) > 0 else None)
    with col3:
        st.metric("Losers (≤-3%)", len(losers), delta=f"-{len(losers)}" if len(losers) > 0 else None, delta_color="inverse")
    
    # Debug info expander
    with st.expander("🔍 Debug Info (News Fetch Status)"):
        news_sources = st.session_state.get('news_fetch_sources', {})
        if news_sources:
            debug_df = pd.DataFrame([
                {"Ticker": t, "News Source": s} 
                for t, s in news_sources.items()
            ])
            st.dataframe(debug_df, use_container_width=True, hide_index=True)
        else:
            st.write("No news fetch data available.")


def render_dividends_tab(engine):
    """Render the Dividends tab."""
    st.subheader("Dividend Tracker")
    
    with st.spinner("Fetching dividend data..."):
        div_data = get_cached_dividend_data(engine)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    total_recorded = div_data['total_recorded']
    total_recent = div_data['total_recent']
    total_all = total_recorded + total_recent
    
    col1.metric("Recorded Dividends (CSV)", f"${total_recorded:,.2f}")
    col2.metric("New Dividends (Post-CSV)", f"${total_recent:,.2f}")
    col3.metric("Total Dividends", f"${total_all:,.2f}")
    
    st.markdown("---")
    
    # Upcoming Dividends
    st.subheader("Upcoming Dividends")
    upcoming = div_data['upcoming_dividends']
    
    if upcoming:
        upcoming_df = pd.DataFrame(upcoming)
        upcoming_df = upcoming_df.rename(columns={
            'ticker': 'Ticker',
            'ex_date': 'Ex-Date',
            'annual_rate': 'Annual Rate',
            'shares': 'Shares Held',
            'expected_quarterly': 'Expected (Est.)',
            'yield': 'Yield %'
        })
        
        display_cols = ['Ticker', 'Ex-Date', 'Shares Held', 'Annual Rate', 'Expected (Est.)', 'Yield %']
        display_cols = [c for c in display_cols if c in upcoming_df.columns]
        
        styler = upcoming_df[display_cols].style.format({
            'Annual Rate': '${:.4f}',
            'Expected (Est.)': '${:,.2f}',
            'Yield %': '{:.2f}%',
            'Shares Held': '{:,.0f}'
        })
        
        st.dataframe(styler, use_container_width=True, hide_index=True)
        
        total_expected = sum(d['expected_quarterly'] for d in upcoming)
        st.info(f"Estimated upcoming dividend income: **${total_expected:,.2f}**")
    else:
        st.caption("No upcoming ex-dividend dates found for your holdings.")
    
    st.markdown("---")
    
    # Recent dividends (post-CSV)
    st.subheader("Recent Dividends (After Dec 13, 2025)")
    recent = div_data['recent_dividends']
    
    if recent:
        recent_df = pd.DataFrame(recent)
        recent_df = recent_df.rename(columns={
            'ticker': 'Ticker',
            'date': 'Payment Date',
            'per_share': 'Per Share',
            'shares': 'Shares',
            'amount': 'Amount (USD)',
            'source': 'Source'
        })
        
        display_cols = ['Ticker', 'Payment Date', 'Per Share', 'Shares', 'Amount (USD)']
        display_cols = [c for c in display_cols if c in recent_df.columns]
        
        styler = recent_df[display_cols].style.format({
            'Per Share': '${:.4f}',
            'Amount (USD)': '${:,.2f}',
            'Shares': '{:,.0f}'
        })
        
        st.dataframe(styler, use_container_width=True, hide_index=True)
    else:
        st.caption("No new dividend payments detected since Dec 13, 2025.")
    
    st.markdown("---")
    
    # Historical dividends from CSV
    st.subheader("Recorded Dividend History (From CSV)")
    recorded = div_data['recorded_dividends']
    
    if recorded:
        recorded_df = pd.DataFrame(recorded)
        recorded_df = recorded_df.rename(columns={
            'ticker': 'Ticker',
            'date': 'Date',
            'amount': 'Amount (USD)',
            'currency': 'Original Currency',
            'shares': 'Shares at Time'
        })
        
        display_cols = ['Ticker', 'Date', 'Amount (USD)', 'Original Currency', 'Shares at Time']
        display_cols = [c for c in display_cols if c in recorded_df.columns]
        
        styler = recorded_df[display_cols].style.format({
            'Amount (USD)': '${:,.2f}',
            'Shares at Time': '{:,.0f}'
        })
        
        st.dataframe(styler, use_container_width=True, hide_index=True)
    else:
        st.caption("No dividend history in transaction records.")
    
    # Combined Dividend View (CSV + Recent)
    st.markdown("---")
    st.subheader("Total Dividend Income by Position")
    
    # Merge CSV dividends with recent yfinance dividends
    combined_dividends = engine.dividend_by_ticker.copy()
    
    # Add recent dividends from the fetch result
    if div_data.get('recent_dividends'):
        for d in div_data['recent_dividends']:
            tk = d['ticker']
            amt = d['amount']
            combined_dividends[tk] = combined_dividends.get(tk, 0.0) + amt
            
    if combined_dividends:
        yield_data = []
        for ticker, divs in combined_dividends.items():
            yield_data.append({
                'Ticker': ticker,
                'Total Dividends': divs
            })
        
        if yield_data:
            yield_df = pd.DataFrame(yield_data).sort_values('Total Dividends', ascending=False)
            
            fig = px.bar(
                yield_df,
                x='Ticker',
                y='Total Dividends',
                color='Total Dividends',
                color_continuous_scale='Greens',
                text_auto='.2s'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                xaxis_title='',
                yaxis_title='Dividends (USD)',
                yaxis_tickprefix='$'
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    if st_autorefresh is not None:
        st_autorefresh(interval=3 * 60 * 1000, key="portfolio_autorefresh")
    
    st.title("Portfolio Tracker")

    with st.sidebar:
        st.caption("Auto-refresh: 3 min")
        st.markdown("### Data source")
        data_mode = st.radio(
            "Select data input",
            ["Use default CSVs", "Upload CSVs"],
            index=0
        )
        uploaded_txn = None
        uploaded_open = None
        if data_mode == "Upload CSVs":
            uploaded_txn = st.file_uploader("Transaction history CSV", type=["csv"], key="txn_upload")
            uploaded_open = st.file_uploader("Open positions CSV (optional)", type=["csv"], key="open_upload")

        st.markdown("### Analytics")
        risk_free_rate_pct = st.number_input(
            "Risk-free rate (annual, %)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.25
        )

        st.markdown("### Benchmark")
        benchmark_choice = st.selectbox(
            "Benchmark ticker",
            ["SPY", "QQQ", "VTI", "IWM", "Custom"],
            index=0
        )
        if benchmark_choice == "Custom":
            benchmark_ticker = st.text_input("Custom benchmark", value="SPY")
        else:
            benchmark_ticker = benchmark_choice

        if st.button("Clear cache & reload"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                st.cache_resource.clear()
            except Exception:
                pass
            st.rerun()

    if data_mode == "Upload CSVs" and uploaded_txn is None:
        st.info("Upload a transaction CSV to load your portfolio.")
        return

    if uploaded_txn is not None:
        txn_path, txn_sig, txn_name = save_uploaded_file(uploaded_txn, "txn")
    else:
        txn_path = os.path.join(BASE_DIR, TRANSACTION_FILE)
        txn_sig = get_file_signature(txn_path)
        txn_name = os.path.basename(txn_path)

    if uploaded_open is not None:
        open_path, open_sig, open_name = save_uploaded_file(uploaded_open, "open")
    else:
        default_open = os.path.join(BASE_DIR, OPEN_POSITION_FILE)
        open_path = default_open if os.path.exists(default_open) else None
        open_sig = get_file_signature(default_open) if open_path else "none"
        open_name = os.path.basename(default_open) if open_path else "None"

    risk_free_rate = risk_free_rate_pct / 100
    benchmark_ticker = (benchmark_ticker or "SPY").strip().upper()

    if not txn_path or not os.path.exists(txn_path):
        st.error("Transaction CSV not found. Please upload a file or verify the default path.")
        return
    
    with st.spinner("Loading..."):
        engine = load_engine(txn_path, open_path, (txn_sig, open_sig))
    
    if not engine:
        return
    
    # Note: fetch_market_data is already called in load_engine() which is cached
    # Don't call it again here or it will cause rate limiting!
    
    valuation_data = engine.get_valuations()
    history_df = get_cached_history(engine)
    timeframe_returns = get_cached_timeframe_returns(engine)
    missing_prices = []
    if getattr(engine, 'market_data', None) is not None and not engine.market_data.empty:
        try:
            missing_prices = engine.market_data[engine.market_data['Price'] <= 0].index.tolist()
        except Exception:
            missing_prices = []

    with st.sidebar:
        st.markdown("### Data status")
        st.caption(f"Transactions: {txn_name}")
        st.caption(f"Open positions: {open_name}")
        if history_df is not None and not history_df.empty:
            st.write(f"History through: {history_df.index.max().strftime('%Y-%m-%d')}")
        st.write(f"Last refresh: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

        stale = getattr(engine, 'stale_tickers', [])
        if stale:
            st.warning(f"Stale prices: {len(stale)} ({', '.join(stale)})")
        else:
            st.caption("No stale prices detected.")

        with st.expander("FX rates used"):
            st.write(engine.fx_rates)

        with st.expander("Data health"):
            warnings = getattr(engine, 'warnings', [])
            if warnings:
                st.warning("\n".join(warnings))
            else:
                st.caption("No data warnings detected.")

            if missing_prices:
                st.error(f"Missing prices for: {', '.join(missing_prices)}")
            else:
                st.caption("All tickers have price data.")
    
    tab_overview, tab_holdings, tab_transactions, tab_summary, tab_dividends, tab_analysis = st.tabs(
        ["Overview", "Holdings", "Transactions", "Summary", "Dividends", "Analysis"]
    )
    
    with tab_overview:
        render_overview_tab(engine, valuation_data, history_df, timeframe_returns)
    
    with tab_holdings:
        render_holdings_tab(engine, history_df)

    with tab_transactions:
        render_transactions_tab(engine)
    
    with tab_summary:
        render_summary_tab()
    
    with tab_dividends:
        render_dividends_tab(engine)
    
    with tab_analysis:
        render_analysis_tab(engine, history_df, risk_free_rate, benchmark_ticker)
    
    # Footer
    st.markdown("---")
    st.caption(
        f"Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Commissions: ${engine.TOTAL_COMMISSIONS:,.2f} | "
        f"Dividends: ${engine.total_dividends:,.2f}"
    )


if __name__ == "__main__":
    main()
