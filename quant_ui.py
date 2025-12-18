"""
Advanced Quantitative Analysis UI Module
========================================
Streamlit components for the quant analysis dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
from scipy import stats

from quant_engine import (
    QuantEngine, 
    TOOLTIPS, 
    get_tooltip_css,
    FactorExposures,
    VolatilityMetrics,
    RiskMetrics,
    DistributionMetrics
)


# =============================================================================
# TOOLTIP HELPER FUNCTIONS
# =============================================================================

def tooltip(key: str) -> str:
    """Create a simple tooltip marker that works with Streamlit."""
    if key not in TOOLTIPS:
        return ""
    tip = TOOLTIPS[key]
    return f" ℹ️"


def render_tooltip_expander(key: str):
    """Render an expandable tooltip explanation."""
    if key not in TOOLTIPS:
        return
    tip = TOOLTIPS[key]
    with st.expander(f"ℹ️ What is {tip['title']}?", expanded=False):
        st.markdown(f"**{tip['explanation']}**")
        st.caption(f"*How to interpret:* {tip['interpretation']}")


def metric_with_help(label: str, value: str, tooltip_key: str, delta: str = None, delta_color: str = "normal"):
    """Display a metric with a help tooltip."""
    col1, col2 = st.columns([10, 1])
    with col1:
        if delta:
            st.metric(label, value, delta, delta_color=delta_color)
        else:
            st.metric(label, value)
    with col2:
        if tooltip_key in TOOLTIPS:
            tip = TOOLTIPS[tooltip_key]
            st.markdown(
                f'<span title="{tip["explanation"]} | {tip["interpretation"]}" '
                f'style="cursor:help; font-size:18px;">❓</span>',
                unsafe_allow_html=True
            )


# =============================================================================
# FACTOR ANALYSIS VISUALIZATIONS
# =============================================================================

def render_factor_analysis(quant: QuantEngine, portfolio_value: float):
    """Render the factor analysis section."""
    st.markdown("## 📊 Multi-Factor Analysis")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 20px; color: white;">
        <strong>What is Factor Analysis?</strong><br>
        We decompose your portfolio's returns into exposures to different market factors 
        (market risk, company size, value vs growth, momentum). This reveals the true 
        drivers of your returns beyond just "the market went up."
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Computing factor exposures..."):
        exposures = quant.compute_factor_exposures()
    
    # Factor Exposure Cards
    st.subheader("Factor Exposures")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "normal" if 0.8 <= exposures.market_beta <= 1.2 else "inverse"
        st.metric(
            "Market Beta ❓",
            f"{exposures.market_beta:.2f}",
            f"{'Higher' if exposures.market_beta > 1 else 'Lower'} than market",
            delta_color=color,
            help=TOOLTIPS['beta']['explanation']
        )
    
    with col2:
        st.metric(
            "Size (SMB) ❓",
            f"{exposures.size_beta:.2f}",
            "Small cap tilt" if exposures.size_beta > 0 else "Large cap tilt",
            help=TOOLTIPS['size_factor']['explanation']
        )
    
    with col3:
        st.metric(
            "Value (HML) ❓",
            f"{exposures.value_beta:.2f}",
            "Value tilt" if exposures.value_beta > 0 else "Growth tilt",
            help=TOOLTIPS['value_factor']['explanation']
        )
    
    with col4:
        st.metric(
            "Momentum ❓",
            f"{exposures.momentum_beta:.2f}",
            "Trend following" if exposures.momentum_beta > 0 else "Contrarian",
            help=TOOLTIPS['momentum_factor']['explanation']
        )
    
    st.markdown("---")
    
    # Performance Metrics
    st.subheader("Risk-Adjusted Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        alpha_color = "normal" if exposures.alpha >= 0 else "inverse"
        st.metric(
            "Jensen's Alpha ❓",
            f"{exposures.alpha*100:.2f}%",
            "Outperforming" if exposures.alpha > 0 else "Underperforming",
            delta_color=alpha_color,
            help=TOOLTIPS['alpha']['explanation']
        )
    
    with col2:
        st.metric(
            "R-Squared ❓",
            f"{exposures.r_squared*100:.1f}%",
            help=TOOLTIPS['r_squared']['explanation']
        )
    
    with col3:
        st.metric(
            "Tracking Error ❓",
            f"{exposures.tracking_error*100:.1f}%",
            help=TOOLTIPS['tracking_error']['explanation']
        )
    
    with col4:
        ir_color = "normal" if exposures.information_ratio > 0 else "inverse"
        st.metric(
            "Information Ratio ❓",
            f"{exposures.information_ratio:.2f}",
            delta_color=ir_color,
            help=TOOLTIPS['information_ratio']['explanation']
        )
    
    # Factor Attribution Chart
    st.subheader("Return Attribution by Factor")
    
    contrib = exposures.factor_contributions
    contrib_df = pd.DataFrame({
        'Factor': list(contrib.keys()),
        'Contribution': [v * 100 for v in contrib.values()]
    })
    
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in contrib_df['Contribution']]
    
    fig = go.Figure(go.Bar(
        x=contrib_df['Factor'],
        y=contrib_df['Contribution'],
        marker_color=colors,
        text=[f"{x:+.2f}%" for x in contrib_df['Contribution']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="<b>Annualized Return Attribution</b>",
        xaxis_title="Factor",
        yaxis_title="Contribution (%)",
        height=400,
        showlegend=False
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("""
    <div style="background: #f0f7ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1E88E5;">
        <strong>📝 How to Read This:</strong><br>
        Each bar shows how much each factor contributed to your returns. 
        <strong>Alpha</strong> is the "skill" component - returns not explained by any factor.
        If Alpha is negative but Market is positive, you're taking market risk but not being rewarded beyond that.
    </div>
    """, unsafe_allow_html=True)
    
    # Rolling Correlations Heatmap
    st.markdown("---")
    st.subheader("📈 Rolling Factor Correlations")
    
    rolling_corrs = quant.compute_rolling_correlations()
    
    if rolling_corrs:
        window_options = list(rolling_corrs.keys())
        selected_window = st.selectbox(
            "Select Rolling Window",
            window_options,
            help="Choose the lookback period for correlation calculation"
        )
        
        if selected_window in rolling_corrs:
            corr_data = rolling_corrs[selected_window]
            
            if not corr_data.empty:
                # Time series of correlations
                fig = go.Figure()
                
                colors = {'MKT': '#3498db', 'SMB': '#e74c3c', 'HML': '#2ecc71', 'MOM': '#9b59b6'}
                
                for col in corr_data.columns:
                    fig.add_trace(go.Scatter(
                        x=corr_data.index,
                        y=corr_data[col],
                        name=col,
                        mode='lines',
                        line=dict(color=colors.get(col, '#333'), width=2)
                    ))
                
                fig.update_layout(
                    title=f"<b>{selected_window} Rolling Correlation with Factors</b>",
                    xaxis_title="Date",
                    yaxis_title="Correlation",
                    height=400,
                    hovermode='x unified',
                    yaxis=dict(range=[-1, 1])
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Current correlation matrix heatmap
                st.markdown("**Current Correlation Matrix**")
                
                current_corrs = corr_data.iloc[-1:].T
                current_corrs.columns = ['Correlation']
                
                fig_heatmap = px.imshow(
                    current_corrs.values.reshape(1, -1),
                    x=current_corrs.index.tolist(),
                    y=['Portfolio'],
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1,
                    text_auto='.2f'
                )
                fig_heatmap.update_layout(height=150)
                
                st.plotly_chart(fig_heatmap, use_container_width=True)


# =============================================================================
# VOLATILITY ANALYSIS VISUALIZATIONS
# =============================================================================

def render_volatility_analysis(quant: QuantEngine, history_df: pd.DataFrame):
    """Render the volatility modeling section."""
    st.markdown("## 📉 Volatility Modeling (GARCH)")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 20px; color: white;">
        <strong>What is GARCH?</strong><br>
        GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models capture 
        "volatility clustering" - the tendency for calm and turbulent periods to persist.
        Unlike simple standard deviation, GARCH predicts <em>future</em> volatility.
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Fitting GARCH model..."):
        vol_metrics = quant.compute_volatility_metrics()
    
    # Volatility Metrics Cards
    st.subheader("Current Volatility Estimates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Historical Volatility ❓",
            f"{vol_metrics.historical_vol:.1f}%",
            "Annualized",
            help=TOOLTIPS['volatility']['explanation']
        )
    
    with col2:
        garch_vs_hist = vol_metrics.garch_vol / vol_metrics.historical_vol - 1
        delta_text = f"{garch_vs_hist*100:+.0f}% vs historical"
        st.metric(
            "GARCH Volatility ❓",
            f"{vol_metrics.garch_vol:.1f}%",
            delta_text,
            delta_color="inverse" if garch_vs_hist > 0.1 else "normal",
            help=TOOLTIPS['garch_volatility']['explanation']
        )
    
    with col3:
        st.metric(
            "EWMA Volatility",
            f"{vol_metrics.ewma_vol:.1f}%",
            "RiskMetrics Model"
        )
    
    st.markdown("---")
    
    # Volatility Forecasts
    st.subheader("📅 Volatility Forecasts")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("1-Day Forecast", f"{vol_metrics.vol_forecast_1d:.1f}%")
    
    with col2:
        st.metric("5-Day Forecast", f"{vol_metrics.vol_forecast_5d:.1f}%")
    
    with col3:
        st.metric("20-Day Forecast", f"{vol_metrics.vol_forecast_20d:.1f}%")
    
    with col4:
        persistence_status = "High" if vol_metrics.persistence > 0.95 else "Moderate"
        st.metric(
            "Persistence ❓",
            f"{vol_metrics.persistence:.3f}",
            persistence_status,
            help=TOOLTIPS['volatility_persistence']['explanation']
        )
    
    # Volatility Cone Chart
    st.markdown("---")
    st.subheader("📊 Volatility Cone")
    
    vol_cone = quant.compute_volatility_cone()
    
    if not vol_cone.empty:
        fig = go.Figure()
        
        # Add percentile bands
        fig.add_trace(go.Scatter(
            x=vol_cone['Horizon'].tolist() + vol_cone['Horizon'].tolist()[::-1],
            y=vol_cone['Max'].tolist() + vol_cone['Min'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='5th-95th Percentile',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=vol_cone['Horizon'].tolist() + vol_cone['Horizon'].tolist()[::-1],
            y=vol_cone['P75'].tolist() + vol_cone['P25'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.4)',
            line=dict(color='rgba(255,255,255,0)'),
            name='25th-75th Percentile',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=vol_cone['Horizon'],
            y=vol_cone['Median'],
            mode='lines+markers',
            name='Median',
            line=dict(color='#2c3e50', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=vol_cone['Horizon'],
            y=vol_cone['Current'],
            mode='lines+markers',
            name='Current',
            line=dict(color='#e74c3c', width=3, dash='dash'),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="<b>Volatility Cone: Historical Range vs Current</b>",
            xaxis_title="Lookback Window (Days)",
            yaxis_title="Annualized Volatility (%)",
            height=450,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
            <strong>📝 How to Read the Volatility Cone:</strong><br>
            The shaded area shows the historical range of volatility at each lookback window.
            The <span style="color:#e74c3c;font-weight:bold;">red dashed line</span> shows where 
            current volatility sits. If it's near the top, markets are unusually turbulent. 
            Near the bottom suggests unusual calm (which often precedes storms!).
        </div>
        """, unsafe_allow_html=True)
    
    # Volatility Time Series (if we have historical portfolio values)
    if not history_df.empty and 'Total' in history_df.columns:
        st.markdown("---")
        st.subheader("📈 Historical Volatility Evolution")
        
        # Calculate rolling volatility
        returns = np.log(history_df['Total'] / history_df['Total'].shift(1)).dropna()
        
        windows = [10, 20, 60]
        fig = go.Figure()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        for i, w in enumerate(windows):
            rolling_vol = returns.rolling(w).std() * np.sqrt(252) * 100
            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name=f'{w}-day',
                line=dict(color=colors[i], width=2)
            ))
        
        fig.update_layout(
            title="<b>Rolling Volatility Over Time</b>",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
            height=350,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MONTE CARLO SIMULATION VISUALIZATIONS
# =============================================================================

def render_monte_carlo(quant: QuantEngine, portfolio_value: float):
    """Render the Monte Carlo simulation section."""
    st.markdown("## 🎲 Monte Carlo Simulation")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 20px; color: white;">
        <strong>What is Monte Carlo Simulation?</strong><br>
        We run thousands of random simulations of your portfolio's future based on 
        historical patterns. This shows the <em>range</em> of possible outcomes, 
        not just a single prediction. We use a Jump-Diffusion model that accounts 
        for sudden market shocks ("black swans").
    </div>
    """, unsafe_allow_html=True)
    
    # Simulation Parameters
    st.subheader("⚙️ Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        horizon = st.slider(
            "Forecast Horizon (Days)",
            min_value=5,
            max_value=252,
            value=63,  # ~3 months
            step=5,
            help="How far into the future to simulate"
        )
    
    with col2:
        n_sims = st.select_slider(
            "Number of Simulations",
            options=[1000, 5000, 10000, 25000],
            value=10000,
            help="More simulations = more accurate but slower"
        )
    
    with col3:
        model = st.radio(
            "Model Type",
            ["jump_diffusion", "gbm"],
            format_func=lambda x: "Jump-Diffusion (Merton)" if x == "jump_diffusion" else "Geometric Brownian Motion",
            help="Jump-diffusion accounts for sudden crashes; GBM assumes smooth movements"
        )
    
    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            results = quant.run_monte_carlo_simulation(
                initial_value=portfolio_value,
                horizon_days=horizon,
                n_simulations=n_sims,
                model=model
            )
        
        # Store in session state
        st.session_state['mc_results'] = results
    
    # Display results if available
    if 'mc_results' in st.session_state:
        results = st.session_state['mc_results']
        stats = results['statistics']
        
        st.markdown("---")
        st.subheader("📊 Simulation Results")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            expected_change = stats['mean_terminal'] - portfolio_value
            st.metric(
                "Expected Value",
                f"${stats['mean_terminal']:,.0f}",
                f"{expected_change:+,.0f} ({stats['expected_return']:+.1f}%)"
            )
        
        with col2:
            st.metric(
                "Median Outcome",
                f"${stats['median_terminal']:,.0f}"
            )
        
        with col3:
            st.metric(
                "Probability of Loss",
                f"{stats['prob_loss']:.1f}%",
                delta_color="inverse" if stats['prob_loss'] > 50 else "normal"
            )
        
        with col4:
            st.metric(
                "Chance of 10%+ Gain",
                f"{stats['prob_gain_10pct']:.1f}%"
            )
        
        # Confidence Interval
        st.markdown("### 📈 Outcome Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Histogram of terminal values
            terminal_values = results['terminal_values']
            
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=terminal_values,
                nbinsx=100,
                name='Simulated Outcomes',
                marker_color='#3498db',
                opacity=0.7
            ))
            
            # Add VaR lines
            var_95 = np.percentile(terminal_values, 5)
            var_99 = np.percentile(terminal_values, 1)
            
            fig.add_vline(x=portfolio_value, line_dash="solid", line_color="green", 
                         annotation_text="Current Value", annotation_position="top")
            fig.add_vline(x=var_95, line_dash="dash", line_color="orange",
                         annotation_text="5th Percentile", annotation_position="top left")
            fig.add_vline(x=var_99, line_dash="dash", line_color="red",
                         annotation_text="1st Percentile", annotation_position="top left")
            fig.add_vline(x=stats['mean_terminal'], line_dash="dot", line_color="purple",
                         annotation_text="Expected", annotation_position="top right")
            
            fig.update_layout(
                title=f"<b>Distribution of Portfolio Values after {results['horizon_days']} Days</b>",
                xaxis_title="Portfolio Value ($)",
                yaxis_title="Frequency",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Summary table
            st.markdown("**Key Percentiles**")
            
            percentile_data = {
                '1st (Worst)': f"${stats['worst_case']:,.0f}",
                '5th': f"${stats['percentile_5']:,.0f}",
                '25th': f"${stats['percentile_25']:,.0f}",
                'Median': f"${stats['median_terminal']:,.0f}",
                '75th': f"${stats['percentile_75']:,.0f}",
                '95th': f"${stats['percentile_95']:,.0f}",
                '99th (Best)': f"${stats['best_case']:,.0f}",
            }
            
            for label, value in percentile_data.items():
                st.markdown(f"**{label}:** {value}")
        
        # Sample Paths Visualization
        st.markdown("### 🛤️ Sample Simulation Paths")
        
        # Show subset of paths
        n_paths_to_show = 100
        paths = results['price_paths'][:n_paths_to_show]
        
        fig = go.Figure()
        
        # Add individual paths with low opacity
        for i in range(min(50, n_paths_to_show)):
            fig.add_trace(go.Scatter(
                y=paths[i],
                mode='lines',
                line=dict(width=0.5, color='rgba(52, 152, 219, 0.15)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add percentile bounds
        percentile_5_path = np.percentile(paths, 5, axis=0)
        percentile_50_path = np.percentile(paths, 50, axis=0)
        percentile_95_path = np.percentile(paths, 95, axis=0)
        
        fig.add_trace(go.Scatter(
            y=percentile_5_path,
            mode='lines',
            name='5th Percentile',
            line=dict(width=2, color='#e74c3c', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            y=percentile_50_path,
            mode='lines',
            name='Median',
            line=dict(width=3, color='#2c3e50')
        ))
        
        fig.add_trace(go.Scatter(
            y=percentile_95_path,
            mode='lines',
            name='95th Percentile',
            line=dict(width=2, color='#27ae60', dash='dash')
        ))
        
        fig.add_hline(y=portfolio_value, line_dash="dot", line_color="gray",
                     annotation_text="Starting Value")
        
        fig.update_layout(
            title=f"<b>Monte Carlo Paths ({n_sims:,} simulations, showing {n_paths_to_show})</b>",
            xaxis_title="Days",
            yaxis_title="Portfolio Value ($)",
            height=450,
            yaxis_tickprefix='$',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Natural Language Insights
        st.markdown("### 💡 Key Insights")
        
        insights = quant.generate_monte_carlo_insights(results, portfolio_value)
        for insight in insights:
            st.markdown(insight)


# =============================================================================
# RISK METRICS VISUALIZATIONS
# =============================================================================

def render_risk_metrics(quant: QuantEngine, portfolio_value: float):
    """Render the VaR and risk metrics section."""
    st.markdown("## ⚠️ Value at Risk & Tail Risk")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <strong>What is Value at Risk (VaR)?</strong><br>
        VaR answers: "What's the most I could lose on a typical bad day?" 
        For example, 95% VaR of $1,000 means you'll lose more than $1,000 only 
        about 1 day per month. <strong>CVaR</strong> tells you how bad it gets 
        when VaR is exceeded.
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Computing risk metrics..."):
        risk = quant.compute_risk_metrics(portfolio_value)
        dist = quant.compute_distribution_metrics()
    
    # VaR Metrics
    st.subheader("📊 Daily Value at Risk")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "95% VaR (Historical) ❓",
            f"${risk.var_95_historical:,.0f}",
            "Max daily loss 19/20 days",
            help=TOOLTIPS['var_95']['explanation']
        )
    
    with col2:
        st.metric(
            "99% VaR (Historical) ❓",
            f"${risk.var_99_historical:,.0f}",
            "Max daily loss 99/100 days",
            help=TOOLTIPS['var_99']['explanation']
        )
    
    with col3:
        st.metric(
            "95% CVaR (ES) ❓",
            f"${risk.cvar_95:,.0f}",
            "Avg loss beyond VaR",
            help=TOOLTIPS['cvar']['explanation']
        )
    
    with col4:
        st.metric(
            "99% CVaR (ES)",
            f"${risk.cvar_99:,.0f}",
            "Extreme tail average"
        )
    
    st.markdown("---")
    
    # Distribution Metrics
    st.subheader("📈 Return Distribution Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Expected Return",
            f"{dist.mean:.1f}%",
            "Annualized"
        )
    
    with col2:
        st.metric(
            "Volatility",
            f"{dist.std:.1f}%",
            "Annualized"
        )
    
    with col3:
        skew_status = "Negative (crash risk)" if dist.skewness < -0.3 else ("Positive" if dist.skewness > 0.3 else "Symmetric")
        st.metric(
            "Skewness ❓",
            f"{dist.skewness:.2f}",
            skew_status,
            delta_color="inverse" if dist.skewness < -0.3 else "normal",
            help=TOOLTIPS['skewness']['explanation']
        )
    
    with col4:
        kurt_status = "Fat tails!" if dist.kurtosis > 4 else "Normal-ish"
        st.metric(
            "Kurtosis ❓",
            f"{dist.kurtosis:.2f}",
            kurt_status + " (Normal=3)",
            delta_color="inverse" if dist.kurtosis > 4 else "off",
            help=TOOLTIPS['kurtosis']['explanation']
        )
    
    # Drawdown and Calmar
    st.markdown("---")
    st.subheader("📉 Drawdown Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Maximum Drawdown ❓",
            f"{risk.max_drawdown:.1f}%",
            "Worst peak-to-trough decline",
            help=TOOLTIPS['max_drawdown']['explanation']
        )
    
    with col2:
        calmar_status = "Good" if risk.calmar_ratio > 1 else "Poor"
        st.metric(
            "Calmar Ratio",
            f"{risk.calmar_ratio:.2f}",
            f"{calmar_status} (Return/MaxDD)"
        )
    
    # Return Distribution Chart with VaR markers
    st.markdown("---")
    st.subheader("📊 Return Distribution with VaR Cutoffs")
    
    returns = quant.portfolio_returns.values * 100  # Convert to percentage
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Daily Returns',
        marker_color='#3498db',
        opacity=0.7
    ))
    
    # Normal distribution overlay
    x_norm = np.linspace(returns.min(), returns.max(), 100)
    y_norm = stats.norm.pdf(x_norm, np.mean(returns), np.std(returns))
    # Scale to match histogram
    y_norm = y_norm * len(returns) * (returns.max() - returns.min()) / 50
    
    fig.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    
    # VaR lines
    var_95_pct = np.percentile(returns, 5)
    var_99_pct = np.percentile(returns, 1)
    
    fig.add_vline(x=var_95_pct, line_dash="dash", line_color="orange",
                 annotation_text=f"95% VaR: {var_95_pct:.2f}%")
    fig.add_vline(x=var_99_pct, line_dash="dash", line_color="red",
                 annotation_text=f"99% VaR: {var_99_pct:.2f}%")
    fig.add_vline(x=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="<b>Daily Return Distribution vs Normal</b>",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Normality test result
    normality_color = "#27ae60" if dist.is_normal else "#e74c3c"
    normality_text = "PASSED" if dist.is_normal else "FAILED"
    
    st.markdown(f"""
    <div style="background: {normality_color}20; padding: 15px; border-radius: 8px; 
                border-left: 4px solid {normality_color};">
        <strong>🧪 Normality Test (Jarque-Bera): {normality_text}</strong><br>
        {'Returns appear normally distributed. Standard VaR estimates are reliable.' if dist.is_normal else
         f'Returns are NOT normally distributed (p-value: {dist.jarque_bera_pvalue:.4f}). The red dashed normal curve underestimates tail risk. Use Historical VaR and CVaR for better risk estimates.'}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# STRESS TESTING VISUALIZATIONS
# =============================================================================

def render_stress_testing(quant: QuantEngine, portfolio_value: float):
    """Render the stress testing section."""
    st.markdown("## 🔥 Stress Testing")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <strong>What is Stress Testing?</strong><br>
        Stress tests show how your portfolio would perform under extreme scenarios 
        like market crashes, rate shocks, or black swan events. It's like a fire drill 
        for your portfolio.
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Stress Sliders
    st.subheader("🎛️ Custom Stress Scenario")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market_shock = st.slider(
            "Market Shock (%)",
            min_value=-50,
            max_value=50,
            value=-10,
            step=1,
            help="Simulate a market-wide move"
        )
    
    with col2:
        vol_multiplier = st.slider(
            "Volatility Multiplier",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="How much volatility increases (1.0 = normal)"
        )
    
    with col3:
        rate_shock = st.slider(
            "Interest Rate Change (%)",
            min_value=-2.0,
            max_value=3.0,
            value=0.0,
            step=0.25,
            help="Change in interest rates"
        )
    
    # Calculate custom scenario impact
    exposures = quant.compute_factor_exposures()
    beta = exposures.market_beta
    
    portfolio_impact = beta * (market_shock / 100) - 0.5 * (rate_shock / 100)  # Simplified
    vol_adjustment = (vol_multiplier - 1) * 0.02 * beta
    total_impact = portfolio_impact - abs(vol_adjustment)
    
    new_value = portfolio_value * (1 + total_impact)
    dollar_impact = new_value - portfolio_value
    
    # Display custom scenario result
    impact_color = "#27ae60" if dollar_impact >= 0 else "#e74c3c"
    
    st.markdown(f"""
    <div style="background: {impact_color}20; padding: 20px; border-radius: 10px; 
                border: 2px solid {impact_color}; margin: 20px 0;">
        <h3 style="margin:0; color: {impact_color};">
            Custom Scenario Impact: ${dollar_impact:+,.0f} ({total_impact*100:+.1f}%)
        </h3>
        <p style="margin: 10px 0 0 0;">
            Portfolio Value: ${portfolio_value:,.0f} → ${new_value:,.0f}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Predefined Scenarios Table
    st.subheader("📋 Predefined Stress Scenarios")
    
    with st.spinner("Running stress tests..."):
        stress_results = quant.run_stress_test(portfolio_value)
    
    # Style the dataframe
    def style_severity(val):
        if val == 'High':
            return 'background-color: #ffcccc; color: #cc0000; font-weight: bold;'
        elif val == 'Medium':
            return 'background-color: #fff3cd; color: #856404;'
        else:
            return 'background-color: #d4edda; color: #155724;'
    
    def style_impact(val):
        if isinstance(val, (int, float)):
            if val < -10:
                return 'color: #cc0000; font-weight: bold;'
            elif val < 0:
                return 'color: #e74c3c;'
            elif val > 10:
                return 'color: #27ae60; font-weight: bold;'
            elif val > 0:
                return 'color: #2ecc71;'
        return ''
    
    styled_df = stress_results.style.format({
        'Portfolio Impact (%)': '{:+.1f}%',
        'Portfolio Impact ($)': '${:+,.0f}',
        'New Value ($)': '${:,.0f}'
    }).map(style_severity, subset=['Severity']).map(style_impact, subset=['Portfolio Impact (%)'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Stress Test Bar Chart
    fig = go.Figure()
    
    colors = ['#e74c3c' if x < 0 else '#27ae60' for x in stress_results['Portfolio Impact (%)']]
    
    fig.add_trace(go.Bar(
        y=stress_results['Scenario'],
        x=stress_results['Portfolio Impact (%)'],
        orientation='h',
        marker_color=colors,
        text=[f"${x:+,.0f}" for x in stress_results['Portfolio Impact ($)']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="<b>Portfolio Impact by Stress Scenario</b>",
        xaxis_title="Portfolio Impact (%)",
        yaxis_title="",
        height=450,
        xaxis=dict(ticksuffix='%')
    )
    fig.add_vline(x=0, line_color="gray", line_dash="dash")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D Sensitivity Surface
    st.markdown("---")
    st.subheader("🌐 3D Sensitivity Surface")
    
    with st.spinner("Computing sensitivity surface..."):
        sensitivity = quant.sensitivity_analysis(portfolio_value)
    
    fig = go.Figure(data=[go.Surface(
        x=sensitivity['market_shocks'],
        y=sensitivity['rate_shocks'],
        z=sensitivity['portfolio_values'],
        colorscale='RdYlGn',
        colorbar=dict(title='Portfolio $')
    )])
    
    fig.update_layout(
        title="<b>Portfolio Value Sensitivity to Market & Rate Shocks</b>",
        scene=dict(
            xaxis_title='Market Shock (%)',
            yaxis_title='Rate Shock (%)',
            zaxis_title='Portfolio Value ($)'
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3;">
        <strong>📝 How to Read the 3D Surface:</strong><br>
        Rotate the chart to explore how your portfolio responds to combinations of 
        market moves (x-axis) and interest rate changes (y-axis). Green peaks show 
        favorable outcomes; red valleys show danger zones. The steeper the surface, 
        the more sensitive your portfolio is to those factors.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN INSIGHTS DASHBOARD
# =============================================================================

def render_quant_insights(quant: QuantEngine, portfolio_value: float):
    """Render the AI-generated insights section."""
    st.markdown("## 🧠 AI Risk Insights")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <strong>Natural Language Risk Analysis</strong><br>
        Our AI analyzes your portfolio's risk characteristics and generates 
        plain-English insights to help you understand what the numbers mean.
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Generating insights..."):
        insights = quant.generate_risk_insights(portfolio_value)
    
    for i, insight in enumerate(insights):
        st.markdown(insight)
        if i < len(insights) - 1:
            st.markdown("")  # Spacing


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_advanced_quant_tab(engine, history_df: pd.DataFrame, portfolio_value: float):
    """
    Main entry point for rendering the Advanced Quantitative Analysis tab.
    
    Args:
        engine: PortfolioEngine instance
        history_df: Historical portfolio values DataFrame
        portfolio_value: Current total portfolio value
    """
    st.markdown("# 🔬 Advanced Quantitative Analysis")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 30px; color: white;">
        <h3 style="margin:0;">Institutional-Grade Risk Analytics</h3>
        <p style="margin: 10px 0 0 0;">
            This module uses the same quantitative frameworks employed by hedge funds and 
            institutional investors. Every metric includes an explanation (look for ❓) 
            to help you understand what it means.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have enough data
    if history_df.empty or 'Total' not in history_df.columns:
        st.warning("⚠️ Insufficient historical data for quantitative analysis. Need at least 30 days of history.")
        return
    
    # Calculate log returns
    portfolio_prices = history_df['Total']
    if len(portfolio_prices) < 30:
        st.warning(f"⚠️ Only {len(portfolio_prices)} days of data available. Need at least 30 for reliable analysis.")
        return
    
    portfolio_returns = np.log(portfolio_prices / portfolio_prices.shift(1)).dropna()
    
    # Get benchmark returns if available
    benchmark_returns = None
    try:
        spy_hist = engine.get_sp500_history()
        if spy_hist is not None and len(spy_hist) > 0:
            spy_returns = np.log(spy_hist / spy_hist.shift(1)).dropna()
            benchmark_returns = spy_returns
    except:
        pass
    
    # Initialize quant engine
    quant = QuantEngine(portfolio_returns, benchmark_returns)
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Factor Analysis",
        "📉 Volatility",
        "🎲 Monte Carlo",
        "⚠️ Risk Metrics",
        "🔥 Stress Test",
        "🧠 AI Insights"
    ])
    
    with tab1:
        render_factor_analysis(quant, portfolio_value)
    
    with tab2:
        render_volatility_analysis(quant, history_df)
    
    with tab3:
        render_monte_carlo(quant, portfolio_value)
    
    with tab4:
        render_risk_metrics(quant, portfolio_value)
    
    with tab5:
        render_stress_testing(quant, portfolio_value)
    
    with tab6:
        render_quant_insights(quant, portfolio_value)
    
    # Footer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer**: These models are based on historical data and statistical assumptions. "
        "Past performance does not guarantee future results. All projections are estimates, not guarantees."
    )
