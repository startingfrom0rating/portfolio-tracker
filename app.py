
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
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


@st.cache_resource
def load_engine():
    """Initialize and load data into the Portfolio Engine."""
    path1 = os.path.join(BASE_DIR, TRANSACTION_FILE)
    path2 = os.path.join(BASE_DIR, OPEN_POSITION_FILE)
    
    eng = PortfolioEngine(path1, path2)
    success = eng.load_data()
    
    if not success:
        st.error(f"Failed to load data: {eng.errors}")
        return None
    
    eng.fetch_market_data()
    return eng


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


def render_holdings_tab(engine, history_df):
    """Render the Holdings tab."""
    st.subheader("Current Holdings")
    
    holdings_df = engine.get_holdings_detail()
    
    if holdings_df.empty:
        st.info("No active positions.")
        return
    
    holdings_df = holdings_df.sort_values('Market Value (USD)', ascending=False)
    
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
    
    st.markdown("---")
    
    # Position Details
    st.subheader("Position Details")
    
    tickers = holdings_df['Ticker'].tolist()
    selected = st.selectbox("Select Position", tickers)
    
    if selected and not history_df.empty and selected in history_df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.line(
                history_df,
                x=history_df.index,
                y=selected,
                title=f'{selected} - Value History'
            )
            fig.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title='',
                yaxis_title='Value (USD)',
                yaxis_tickprefix='$'
            )
            fig.update_traces(line_color='#1976d2')
            st.plotly_chart(fig, use_container_width=True)
        
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


def render_analysis_tab(engine, history_df):
    """Render the Analysis tab."""
    
    # Benchmark Comparison
    st.subheader("Performance vs Benchmark")
    
    benchmark = engine.get_benchmark_comparison()
    
    if 'error' in benchmark:
        st.warning(f"Benchmark data unavailable: {benchmark['error']}")
    else:
        port_ret = as_float(benchmark.get('portfolio_return', 0))
        sp_ret = as_float(benchmark.get('sp500_return', 0))
        excess = as_float(benchmark.get('excess_return', 0))
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Portfolio", f"{port_ret:+.2f}%", delta="Since Sep 28", delta_color="off")
        col2.metric("SPY (S&P 500)", f"{sp_ret:+.2f}%", delta="Since Sep 28", delta_color="off")
        col3.metric(
            "Excess Return",
            f"{excess:+.2f}%",
            delta="Outperforming" if excess >= 0 else "Underperforming",
            delta_color="normal" if excess >= 0 else "inverse"
        )
    
    # Benchmark Chart
    if not history_df.empty:
        start_dt = engine.INCEPTION_DATE.normalize() if hasattr(engine, "INCEPTION_DATE") else pd.Timestamp('2025-09-28')
        hist_slice = history_df[history_df.index >= start_dt]
        spy_hist = engine.get_sp500_history(start_date=start_dt)
        
        chart_rendered = False
        
        if spy_hist is not None and len(spy_hist) > 0 and not hist_slice.empty:
            if isinstance(spy_hist, pd.DataFrame):
                spy_hist = spy_hist.iloc[:, 0]
            
            common_dates = hist_slice.index.intersection(spy_hist.index)
            
            if len(common_dates) > 1:
                base_date = common_dates[0]
                port_base = as_float(hist_slice.loc[base_date, 'Total'], default=1.0)
                sp_base = as_float(spy_hist.loc[base_date], default=1.0)
                
                if port_base > 0 and sp_base > 0:
                    port_ret_series = ((hist_slice.loc[common_dates, 'Total'] / port_base) - 1.0) * 100
                    spy_ret_series = ((spy_hist.loc[common_dates] / sp_base) - 1.0) * 100
                    
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
                        x=spy_ret_series.index,
                        y=spy_ret_series.values,
                        name='SPY',
                        line=dict(color='#757575', width=2, dash='dot'),
                    ))
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="#bdbdbd", opacity=0.5)
                    
                    fig.update_layout(
                        title='Cumulative Return: Portfolio vs SPY',
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


def render_dividends_tab(engine):
    """Render the Dividends tab."""
    st.subheader("Dividend Tracker")
    
    with st.spinner("Fetching dividend data..."):
        div_data = engine.get_dividend_data()
    
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
    
    # Dividend yield summary by position
    st.markdown("---")
    st.subheader("Dividend Yield by Position")
    
    dividend_by_ticker = engine.dividend_by_ticker
    if dividend_by_ticker:
        yield_data = []
        for ticker, divs in dividend_by_ticker.items():
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
    
    with st.spinner("Loading..."):
        engine = load_engine()
    
    if not engine:
        return
    
    try:
        engine.fetch_market_data()
    except Exception as e:
        st.warning(f"Price refresh failed. ({e})")
    
    valuation_data = engine.get_valuations()
    history_df = engine.get_history(breakdown=True)
    timeframe_returns = engine.get_timeframe_returns()
    
    tab_overview, tab_holdings, tab_dividends, tab_analysis = st.tabs(["Overview", "Holdings", "Dividends", "Analysis"])
    
    with tab_overview:
        render_overview_tab(engine, valuation_data, history_df, timeframe_returns)
    
    with tab_holdings:
        render_holdings_tab(engine, history_df)
    
    with tab_dividends:
        render_dividends_tab(engine)
    
    with tab_analysis:
        render_analysis_tab(engine, history_df)
    
    # Footer
    st.markdown("---")
    st.caption(
        f"Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Commissions: ${engine.TOTAL_COMMISSIONS:,.2f} | "
        f"Dividends: ${engine.total_dividends:,.2f}"
    )


if __name__ == "__main__":
    main()
