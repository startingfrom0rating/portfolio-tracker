
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from portfolio_engine import PortfolioEngine

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None

# Set Page Config
st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="📈",
    layout="wide"
)

# Constants
TRANSACTION_FILE = "attachment;filename=TransactionHistory_12_13_2025.csv"
OPEN_POSITION_FILE = "attachment;filename=OpenPosition_12_14_2025.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def format_change(value, is_pct=False):
    """Format a change value with color."""
    if is_pct:
        return f"{value:+.2f}%"
    return f"${value:+,.2f}"


def get_color(value):
    """Return color based on positive/negative value."""
    return "green" if value >= 0 else "red"


def as_float(x, default=0.0):
    """Coerce pandas/numpy scalars (or 1-element Series) to python float."""
    try:
        if isinstance(x, pd.Series):
            if len(x) == 0:
                return float(default)
            return float(x.iloc[-1])
        return float(x)
    except Exception:
        return float(default)


def _robust_symmetric_range(values, fallback=1.0, q=0.95):
    """Compute a symmetric +/- range for color scaling from a numeric series."""
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
    """Initializes and loads data into the Portfolio Engine."""
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
    """Render the Overview tab with key metrics and charts."""
    total_val = valuation_data['total_value']
    positions_df = valuation_data['positions']
    
    # --- Key Metrics Row ---
    st.subheader("📊 Portfolio Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Current Value
    col1.metric(
        "Portfolio Value",
        f"${total_val:,.2f}",
        f"{timeframe_returns.get('1D', {}).get('percent', 0):+.2f}% (1D)"
    )
    
    # Net Return (Since Inception, after commissions)
    inception_net = timeframe_returns.get('Inception_Net', {})
    net_return = inception_net.get('absolute', 0)
    net_return_pct = inception_net.get('percent', 0)
    col2.metric(
        "Net Return (Incept.)",
        f"${net_return:,.2f}",
        f"{net_return_pct:+.2f}%"
    )
    
    # Total Dividends
    col3.metric(
        "Total Dividends",
        f"${engine.total_dividends:,.2f}"
    )
    
    # Total Commissions
    col4.metric(
        "Total Commissions",
        f"${engine.TOTAL_COMMISSIONS:,.2f}"
    )
    
    # Cash
    col5.metric(
        "Cash Balance",
        f"${valuation_data['cash']:,.2f}"
    )
    
    st.markdown("---")
    
    # --- Returns Table ---
    st.subheader("📈 Performance Returns (Total Return)")
    
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
        
        # Style the dataframe
        def color_returns(val):
            if isinstance(val, (int, float)):
                color = 'color: green' if val >= 0 else 'color: red'
                return color
            return ''
        
        styled_returns = returns_df.style.format({
            'Return ($)': '${:+,.2f}',
            'Return (%)': '{:+.2f}%'
        }).map(color_returns, subset=['Return ($)', 'Return (%)'])
        
        st.dataframe(styled_returns, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # --- Portfolio Growth Chart ---
    st.subheader("Portfolio Growth (Since Inception)")
    if not history_df.empty:
        fig_hist = px.line(
            history_df, 
            x=history_df.index, 
            y='Total',
            title='Total Portfolio Value (USD)',
            labels={'Total': 'Value ($)', 'index': 'Date'}
        )
        fig_hist.update_layout(hovermode='x unified')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # --- Allocation Charts ---
    col_alloc, col_pnl = st.columns(2)
    
    with col_alloc:
        st.subheader("Asset Allocation")
        if not positions_df.empty:
            fig_tree = px.treemap(
                positions_df,
                path=[px.Constant("Portfolio"), 'Ticker'],
                values='Market Value (USD)',
                color='Unrealized PnL',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                hover_data=['Net Shares', 'Price (Local)', 'Avg Cost', 'PnL %'],
                title="Market Value & PnL Heatmap"
            )
            fig_tree.update_traces(textinfo="label+value+percent parent")
            fig_tree.update_layout(height=450, margin=dict(t=30, l=10, r=10, b=10))
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
                hover_data=['Net Shares', 'Avg Cost', 'Price (Local)', 'PnL %'],
                text_auto='.2s',
                title="Gain/Loss Leaders"
            )
            fig_pnl.update_traces(textposition='outside')
            fig_pnl.update_layout(height=450)
            st.plotly_chart(fig_pnl, use_container_width=True)


def render_holdings_tab(engine, history_df):
    """Render the Holdings tab with detailed position breakdown."""
    st.subheader("📋 Current Holdings")
    
    holdings_df = engine.get_holdings_detail()
    
    if holdings_df.empty:
        st.info("No active positions.")
        return
    
    # Sort by market value
    holdings_df = holdings_df.sort_values('Market Value (USD)', ascending=False)
    
    # Select columns to display
    display_cols = [
        'Ticker', 'Net Shares', 'Price (Local)', 'Currency',
        'Market Value (USD)', 'Avg Cost', 'Unrealized PnL', 'PnL %',
        'Daily Change ($)', 'Daily Change (%)', 'Weight (%)', 'Dividends Received'
    ]
    
    # Filter to existing columns
    display_cols = [c for c in display_cols if c in holdings_df.columns]
    display_df = holdings_df[display_cols].copy()
    
    # Formatting
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
    
    # Apply formatting only to columns that exist
    active_format = {k: v for k, v in format_dict.items() if k in display_df.columns}
    
    # Pandas Styler with gradient
    styler = display_df.style.format(active_format)
    
    try:
        import matplotlib  # noqa: F401
        # Dynamic color scales to avoid "everything is the same color" and handle big movers.
        if 'Unrealized PnL' in display_df.columns:
            v = _robust_symmetric_range(display_df['Unrealized PnL'], fallback=5000.0, q=0.95)
            styler = styler.background_gradient(
                subset=['Unrealized PnL'],
                cmap="RdBu",
                vmin=-v,
                vmax=v,
            )
        if 'Unrealized PnL' in display_df.columns:
            # already applied above
            pass
        if 'Daily Change ($)' in display_df.columns:
            v = _robust_symmetric_range(display_df['Daily Change ($)'], fallback=500.0, q=0.95)
            styler = styler.background_gradient(
                subset=['Daily Change ($)'],
                cmap="RdBu",
                vmin=-v,
                vmax=v,
            )
        # QoL: weight bars are readable and don't require matplotlib.
        if 'Weight (%)' in display_df.columns:
            styler = styler.bar(subset=['Weight (%)'], color='#4e79a7', vmin=0)
    except Exception:
        pass
    
    st.dataframe(styler, use_container_width=True, height=600, hide_index=True)
    
    # --- Per-Equity Drill-Down ---
    st.markdown("---")
    st.subheader("📊 Per-Equity Analysis")
    
    tickers = holdings_df['Ticker'].tolist()
    selected = st.selectbox("Select Position for Details", tickers)
    
    if selected and not history_df.empty and selected in history_df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.line(
                history_df,
                x=history_df.index,
                y=selected,
                title=f'{selected} - Position Value History (USD)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            pos_data = holdings_df[holdings_df['Ticker'] == selected].iloc[0]
            st.markdown(f"**{selected}**")
            st.write(f"Shares: {pos_data['Net Shares']:,.2f}")
            st.write(f"Market Value: ${pos_data['Market Value (USD)']:,.2f}")
            st.write(f"Avg Cost: ${pos_data['Avg Cost']:,.2f}")
            st.write(f"Unrealized P&L: ${pos_data['Unrealized PnL']:+,.2f} ({pos_data['PnL %']:+.2f}%)")
            if 'Weight (%)' in pos_data:
                st.write(f"Portfolio Weight: {pos_data['Weight (%)']:.1f}%")
            if 'Dividends Received' in pos_data:
                st.write(f"Dividends: ${pos_data['Dividends Received']:,.2f}")
            
            # Fetch news (best effort)
            st.markdown("**Recent News:**")
            news = engine.fetch_stock_news(selected.replace('.TO', '').replace('.T', '').replace('.L', ''))
            if news:
                for n in news:
                    st.markdown(f"- [{n['title'][:60]}...]({n['link']})")
            else:
                st.caption("No recent news available.")


def render_analysis_tab(engine, history_df):
    """Render the Analysis tab with benchmarking and attribution."""
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BENCHMARK COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📈 Performance vs Benchmark")
    
    benchmark = engine.get_benchmark_comparison()
    
    if 'error' in benchmark:
        st.warning(f"Benchmark data unavailable: {benchmark['error']}")
    else:
        port_ret = as_float(benchmark.get('portfolio_return', 0))
        sp_ret = as_float(benchmark.get('sp500_return', 0))
        excess = as_float(benchmark.get('excess_return', 0))
        
        # Metrics in a clean card-like layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📊 Portfolio",
                f"{port_ret:+.2f}%",
                delta=f"Since Sep 28",
                delta_color="off"
            )
        
        with col2:
            st.metric(
                "📉 SPY (S&P 500)",
                f"{sp_ret:+.2f}%",
                delta=f"Since Sep 28",
                delta_color="off"
            )
        
        with col3:
            delta_color = "normal" if excess >= 0 else "inverse"
            st.metric(
                "⚡ Alpha (Excess Return)",
                f"{excess:+.2f}%",
                delta=f"{'Outperforming' if excess >= 0 else 'Underperforming'}",
                delta_color=delta_color
            )
    
    # Comparison Chart
    if not history_df.empty:
        start_dt = engine.INCEPTION_DATE.normalize() if hasattr(engine, "INCEPTION_DATE") else pd.Timestamp('2025-09-28')
        hist_slice = history_df[history_df.index >= start_dt]
        
        spy_hist = engine.get_sp500_history(start_date=start_dt)
        
        # Debug info in expander
        chart_rendered = False
        
        if spy_hist is not None and len(spy_hist) > 0 and not hist_slice.empty:
            # Ensure spy_hist is a Series
            if isinstance(spy_hist, pd.DataFrame):
                spy_hist = spy_hist.iloc[:, 0]
            
            common_dates = hist_slice.index.intersection(spy_hist.index)
            
            if len(common_dates) > 1:
                base_date = common_dates[0]
                port_base = as_float(hist_slice.loc[base_date, 'Total'], default=1.0)
                sp_base = as_float(spy_hist.loc[base_date], default=1.0)
                
                if port_base > 0 and sp_base > 0:
                    # Calculate cumulative returns (%)
                    port_ret_series = ((hist_slice.loc[common_dates, 'Total'] / port_base) - 1.0) * 100
                    spy_ret_series = ((spy_hist.loc[common_dates] / sp_base) - 1.0) * 100
                    
                    fig = go.Figure()
                    
                    # Portfolio line
                    fig.add_trace(go.Scatter(
                        x=port_ret_series.index,
                        y=port_ret_series.values,
                        name='Portfolio',
                        line=dict(color='#2962ff', width=2.5),
                        fill='tozeroy',
                        fillcolor='rgba(41, 98, 255, 0.1)',
                        hovertemplate='Portfolio: %{y:+.2f}%<extra></extra>'
                    ))
                    
                    # SPY line
                    fig.add_trace(go.Scatter(
                        x=spy_ret_series.index,
                        y=spy_ret_series.values,
                        name='SPY',
                        line=dict(color='#ff6d00', width=2.5, dash='dot'),
                        hovertemplate='SPY: %{y:+.2f}%<extra></extra>'
                    ))
                    
                    # Zero line
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    fig.update_layout(
                        title=dict(
                            text='<b>Cumulative Return: Portfolio vs SPY</b>',
                            font=dict(size=16)
                        ),
                        xaxis_title='',
                        yaxis_title='Return (%)',
                        yaxis=dict(ticksuffix='%', zeroline=True),
                        hovermode='x unified',
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='right',
                            x=1
                        ),
                        margin=dict(l=0, r=0, t=60, b=0),
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    chart_rendered = True
        
        if not chart_rendered:
            st.info("📊 Benchmark chart unavailable - insufficient overlapping data between portfolio and SPY.")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DAILY PERFORMANCE REVIEW
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📅 Today's Performance")
    
    daily = engine.get_daily_attribution()
    
    if abs(daily.get('total_change', 0)) > 0.01:
        total_chg = daily['total_change']
        total_pct = daily['total_change_pct']
        is_positive = total_chg >= 0
        
        # Summary card
        chg_color = "#00c853" if is_positive else "#ff1744"
        arrow = "▲" if is_positive else "▼"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {'#e8f5e9' if is_positive else '#ffebee'} 0%, white 100%); 
                    padding: 20px; border-radius: 10px; border-left: 4px solid {chg_color}; margin-bottom: 20px;">
            <h3 style="margin: 0; color: {chg_color};">
                {arrow} ${abs(total_chg):,.2f} ({total_pct:+.2f}%)
            </h3>
            <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">Day over Day Change</p>
        </div>
        """, unsafe_allow_html=True)
        
        contributors = daily.get('contributors', [])
        if contributors:
            # Split into winners and losers
            winners = sorted([c for c in contributors if c['change_usd'] > 0], 
                           key=lambda x: x['change_usd'], reverse=True)[:5]
            losers = sorted([c for c in contributors if c['change_usd'] < 0], 
                          key=lambda x: x['change_usd'])[:5]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🟢 Top Gainers")
                if winners:
                    for i, w in enumerate(winners, 1):
                        pct_of_move = abs(w['contribution_pct']) if w['contribution_pct'] else 0
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; 
                                padding: 8px 12px; margin: 4px 0; background: #f1f8e9; border-radius: 6px;">
                            <span style="font-weight: 600; color: black;">{w['ticker']}</span>
                            <span style="color: black; font-weight: 500;">+${w['change_usd']:,.0f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No gainers today")
            
            with col2:
                st.markdown("#### 🔴 Top Losers")
                if losers:
                    for i, l in enumerate(losers, 1):
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; 
                                    padding: 8px 12px; margin: 4px 0; background: #ffebee; border-radius: 6px;">
                            <span style="font-weight: 600; color: black;">{l['ticker']}</span>
                            <span style="color: black; font-weight: 500;">${l['change_usd']:,.0f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No losers today")
            
            # Key insight
            if winners or losers:
                top_mover = winners[0] if winners and (not losers or abs(winners[0]['change_usd']) >= abs(losers[0]['change_usd'])) else (losers[0] if losers else None)
                if top_mover:
                    direction = "drove gains" if top_mover['change_usd'] > 0 else "weighed on performance"
                    st.markdown(f"""
                    <div style="background: #f5f5f5; padding: 12px 16px; border-radius: 8px; margin-top: 16px;">
                        <strong>💡 Key Insight:</strong> {top_mover['ticker']} {direction} with a 
                        ${abs(top_mover['change_usd']):,.0f} move.
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("📊 Market closed or no significant movement today.")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEEKLY PERFORMANCE REVIEW
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📆 This Week's Performance")
    
    weekly = engine.get_weekly_attribution()
    
    if abs(weekly.get('total_change', 0)) > 0.01:
        total_chg = weekly['total_change']
        total_pct = weekly['total_change_pct']
        is_positive = total_chg >= 0
        
        chg_color = "#00c853" if is_positive else "#ff1744"
        arrow = "▲" if is_positive else "▼"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {'#e8f5e9' if is_positive else '#ffebee'} 0%, white 100%); 
                    padding: 20px; border-radius: 10px; border-left: 4px solid {chg_color}; margin-bottom: 20px;">
            <h3 style="margin: 0; color: {chg_color};">
                {arrow} ${abs(total_chg):,.2f} ({total_pct:+.2f}%)
            </h3>
            <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">Week over Week Change</p>
        </div>
        """, unsafe_allow_html=True)
        
        contributors = weekly.get('contributors', [])
        if contributors:
            # Create a horizontal bar chart for weekly attribution
            contrib_df = pd.DataFrame(contributors)
            contrib_df = contrib_df.sort_values('change_usd', key=lambda x: x.abs(), ascending=True).tail(10)
            
            colors = ['#00c853' if x >= 0 else '#ff1744' for x in contrib_df['change_usd']]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=contrib_df['ticker'],
                x=contrib_df['change_usd'],
                orientation='h',
                marker_color=colors,
                text=[f"${x:+,.0f}" for x in contrib_df['change_usd']],
                textposition='outside',
                hovertemplate='%{y}: $%{x:,.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text='<b>Weekly Attribution by Position</b>', font=dict(size=14)),
                xaxis_title='Impact ($)',
                yaxis_title='',
                height=max(300, len(contrib_df) * 35),
                margin=dict(l=0, r=60, t=40, b=40),
                showlegend=False
            )
            
            fig.add_vline(x=0, line_color="gray", line_dash="dash", opacity=0.5)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly narrative
            winners = [c for c in contributors if c['change_usd'] > 0]
            losers = [c for c in contributors if c['change_usd'] < 0]
            
            narrative_parts = []
            if winners:
                top_winners = sorted(winners, key=lambda x: x['change_usd'], reverse=True)[:2]
                names = " and ".join([w['ticker'] for w in top_winners])
                narrative_parts.append(f"**{names}** led performance this week")
            if losers:
                top_losers = sorted(losers, key=lambda x: x['change_usd'])[:2]
                names = " and ".join([l['ticker'] for l in top_losers])
                narrative_parts.append(f"**{names}** were the main detractors")
            
            if narrative_parts:
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 12px 16px; border-radius: 8px; margin-top: 8px;">
                    <strong>📝 Weekly Summary:</strong> {', while '.join(narrative_parts)}.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📊 No significant portfolio movement this week.")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEWS & CATALYSTS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📰 Market News for Top Holdings")
    
    # Get top 5 holdings by weight
    holdings = engine.get_holdings_detail()
    if not holdings.empty:
        top_holdings = holdings.nlargest(5, 'Market Value (USD)')['Ticker'].tolist()
        
        cols = st.columns(min(3, len(top_holdings)))
        for i, ticker in enumerate(top_holdings[:3]):
            with cols[i]:
                clean_ticker = ticker.replace('.TO', '').replace('.T', '').replace('.L', '')
                st.markdown(f"**{ticker}**")
                news = engine.fetch_stock_news(clean_ticker, limit=2)
                if news:
                    for n in news:
                        title = n['title'][:50] + "..." if len(n['title']) > 50 else n['title']
                        st.markdown(f"[{title}]({n['link']})", unsafe_allow_html=True)
                else:
                    st.caption("No recent news")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ASSET DRILL-DOWN
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🔍 Asset Performance Drill-Down")
    
    if not history_df.empty:
        tickers = ['Total', 'Cash'] + sorted([c for c in history_df.columns if c not in ['Total', 'Cash']])
        sel_ticker = st.selectbox("Select asset to analyze", tickers, key="analysis_drilldown")
        
        if sel_ticker:
            fig = px.area(
                history_df,
                x=history_df.index,
                y=sel_ticker,
                title=f'{sel_ticker} - Value Over Time',
                color_discrete_sequence=['#2962ff']
            )
            fig.update_layout(
                xaxis_title='',
                yaxis_title='Value (USD)',
                yaxis_tickprefix='$',
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    # Auto-refresh every 3 minutes
    if st_autorefresh is not None:
        st_autorefresh(interval=3 * 60 * 1000, key="portfolio_autorefresh")
    
    st.title("📈 Portfolio Tracker")

    with st.sidebar:
        st.caption("Auto-refresh: every 3 minutes")
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
    
    # Load Engine
    with st.spinner("Loading Data & Prices..."):
        engine = load_engine()
    
    if not engine:
        return
    
    # Refresh prices
    try:
        engine.fetch_market_data()
    except Exception as e:
        st.warning(f"Price refresh failed; showing last known prices. ({e})")
    
    # Get core data
    valuation_data = engine.get_valuations()
    history_df = engine.get_history(breakdown=True)
    timeframe_returns = engine.get_timeframe_returns()
    
    # --- Tabs ---
    tab_overview, tab_holdings, tab_analysis = st.tabs([
        "📊 Overview",
        "📋 Holdings",
        "🔎 Analysis"
    ])
    
    with tab_overview:
        render_overview_tab(engine, valuation_data, history_df, timeframe_returns)
    
    with tab_holdings:
        render_holdings_tab(engine, history_df)
    
    with tab_analysis:
        render_analysis_tab(engine, history_df)
    
    # Footer
    st.markdown("---")
    st.caption(
        f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Commissions: ${engine.TOTAL_COMMISSIONS:,.2f} | "
        f"Dividends: ${engine.total_dividends:,.2f}"
    )


if __name__ == "__main__":
    main()
