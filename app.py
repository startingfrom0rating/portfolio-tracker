
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from portfolio_engine import PortfolioEngine

# Set Page Config
st.set_page_config(
    page_title="Portfolio Valuation",
    page_icon="📈",
    layout="wide"
)

# Constants
TRANSACTION_FILE = "attachment;filename=TransactionHistory_12_13_2025.csv"
OPEN_POSITION_FILE = "attachment;filename=OpenPosition_12_14_2025.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        
    # Fetch Data immediately upon load
    eng.fetch_market_data()
    return eng

def main():
    st.title("📈 Portfolio Tracker & Valuation")
    
    # Load Engine
    with st.spinner("Loading Data & Prices..."):
        engine = load_engine()

    if not engine:
        return

    # Get Valuations
    valuation_data = engine.get_valuations()
    total_val = valuation_data['total_value']
    cash = valuation_data['cash']
    equity = valuation_data['total_equity']
    positions_df = valuation_data['positions']
    
    # Calculate PnL (Total)
    initial_capital = 500000.0
    total_pnl = total_val - initial_capital
    pnl_pct = (total_pnl / initial_capital) * 100
    
    # Daily Change (needs history)
    history_df = engine.get_history(breakdown=True)
    day_change_val = 0.0
    day_change_pct = 0.0
    if not history_df.empty and len(history_df) > 1:
        today_val = history_df['Total'].iloc[-1]
        yesterday_val = history_df['Total'].iloc[-2]
        day_change_val = today_val - yesterday_val
        day_change_pct = (day_change_val / yesterday_val) * 100
    
    # --- Top Metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Portfolio Value", f"${total_val:,.2f}", f"{day_change_pct:+.2f}% (Day)")
    c2.metric("Total PnL (Incept.)", f"${total_pnl:,.2f}", f"{pnl_pct:+.2f}%")
    c3.metric("Equity Value", f"${equity:,.2f}")
    c4.metric("Cash Balance", f"${cash:,.2f}")
    
    st.markdown("---")
    
    # --- Tabs ---
    tab_overview, tab_analysis, tab_assets = st.tabs(["📊 Overview", "🔎 Analysis", "📋 Holdings"])
    
    with tab_overview:
        # History Chart
        st.subheader("Portfolio Growth")
        if not history_df.empty:
            fig_hist = px.line(history_df, x=history_df.index, y='Total', title='Total Portfolio Value (USD)')
            st.plotly_chart(fig_hist, use_container_width=True)
            
        col_alloc, col_pnl = st.columns(2)
        
        with col_alloc:
            st.subheader("Asset Allocation")
            if not positions_df.empty:
                # TreeMap
                # Enhance with Hover Data
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
                fig_tree.update_layout(height=500, margin=dict(t=30, l=10, r=10, b=10))
                st.plotly_chart(fig_tree, use_container_width=True)
        
        with col_pnl:
            st.subheader("Unrealized PnL by Asset")
            if not positions_df.empty:
                # Bar Chart
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
                fig_pnl.update_layout(height=500)
                st.plotly_chart(fig_pnl, use_container_width=True)

    with tab_analysis:
        st.subheader("Asset Performance Drill-Down")
        tix = ['Total', 'Cash'] + sorted([c for c in history_df.columns if c not in ['Total', 'Cash']])
        sel_ticker = st.selectbox("Select Asset to View History", tix)
        
        if sel_ticker and not history_df.empty:
            fig_asset = px.line(history_df, x=history_df.index, y=sel_ticker, title=f'{sel_ticker} Value History (USD)')
            st.plotly_chart(fig_asset, use_container_width=True)

    with tab_assets:
        st.subheader("Net Positions")
        if not positions_df.empty:
            display_df = positions_df.copy()
            # Sort
            display_df = display_df.sort_values('Market Value (USD)', ascending=False)
            
            # Formatting
            format_dict = {
                "Net Shares": "{:,.2f}",
                "Price (Local)": "{:,.2f}",
                "Market Value (USD)": "${:,.2f}",
                "Avg Cost": "${:,.2f}",
                "Unrealized PnL": "${:,.2f}",
                "PnL %": "{:+.2f}%"
            }
            
            # Pandas Styler
            st.dataframe(
                display_df.style.format(format_dict)
                .background_gradient(subset=['Unrealized PnL'], cmap="RdBu", vmin=-5000, vmax=5000),
                use_container_width=True,
                height=800
            )
        else:
            st.info("No active positions.")
                
if __name__ == "__main__":
    main()
