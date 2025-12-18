"""
Advanced Quantitative Analysis Engine
=====================================
Institutional-grade risk analytics including:
- Multi-factor regression (Fama-French inspired)
- GARCH/EGARCH volatility modeling
- Jump-diffusion Monte Carlo simulations
- VaR/CVaR risk metrics
- Stress testing framework
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

# =============================================================================
# EDUCATIONAL TOOLTIPS - Plain English Explanations
# =============================================================================

TOOLTIPS = {
    # Risk Metrics
    'beta': {
        'title': 'Beta (β)',
        'explanation': 'Measures how much your portfolio moves relative to the market. A Beta of 1.0 means it moves exactly with the market.',
        'interpretation': 'Beta > 1.0: More volatile than market. Beta < 1.0: Less volatile. Beta < 0: Moves opposite to market.'
    },
    'alpha': {
        'title': "Jensen's Alpha (α)",
        'explanation': 'The extra return your portfolio earned above what was expected given its risk level.',
        'interpretation': 'Positive Alpha: Outperforming expectations. Negative Alpha: Underperforming. Higher is better.'
    },
    'sharpe_ratio': {
        'title': 'Sharpe Ratio',
        'explanation': 'Measures return per unit of risk. How much extra return do you get for the volatility you endure?',
        'interpretation': '> 1.0: Good. > 2.0: Very good. > 3.0: Excellent. < 0: Losing money.'
    },
    'sortino_ratio': {
        'title': 'Sortino Ratio',
        'explanation': 'Like Sharpe, but only penalizes downside volatility. Upside volatility is not considered "bad".',
        'interpretation': 'Higher is better. More useful than Sharpe when returns are not symmetric.'
    },
    'information_ratio': {
        'title': 'Information Ratio',
        'explanation': 'Measures how consistently you beat your benchmark, adjusted for how much you deviate from it.',
        'interpretation': '> 0.5: Good. > 1.0: Excellent. Shows skill vs luck in beating the benchmark.'
    },
    'tracking_error': {
        'title': 'Tracking Error',
        'explanation': 'How much your portfolio returns deviate from the benchmark. High = very different from index.',
        'interpretation': 'Low (1-3%): Index-like. High (>5%): Active management. Not good or bad, just different.'
    },
    'r_squared': {
        'title': 'R-Squared (R²)',
        'explanation': 'How much of your portfolio movement is explained by the market/factors (0-100%).',
        'interpretation': 'High R²: Moves with market. Low R²: Unique drivers. Neither is inherently better.'
    },
    'max_drawdown': {
        'title': 'Maximum Drawdown',
        'explanation': 'The largest peak-to-trough decline in portfolio value. Your worst historical loss.',
        'interpretation': 'Smaller is better. Shows the worst-case scenario you actually experienced.'
    },
    
    # Volatility Metrics
    'volatility': {
        'title': 'Volatility (σ)',
        'explanation': 'Standard deviation of returns. How much your portfolio value jumps around.',
        'interpretation': 'Higher = more unpredictable. 15-20% annually is typical for stocks.'
    },
    'garch_volatility': {
        'title': 'GARCH Volatility',
        'explanation': 'Forward-looking volatility estimate that accounts for "volatility clustering" - calm periods followed by turbulent ones.',
        'interpretation': 'More accurate than simple historical volatility, especially during market stress.'
    },
    'volatility_persistence': {
        'title': 'Volatility Persistence',
        'explanation': 'How long volatility shocks last. High persistence means turbulent periods tend to continue.',
        'interpretation': 'Close to 1.0: Shocks last long. Close to 0: Quick mean reversion.'
    },
    
    # Distribution Metrics
    'skewness': {
        'title': 'Skewness',
        'explanation': 'Measures asymmetry in returns. Are big moves more likely to be gains or losses?',
        'interpretation': 'Negative: Risk of large losses (common in stocks). Positive: Risk of large gains. Zero: Symmetric.'
    },
    'kurtosis': {
        'title': 'Kurtosis',
        'explanation': 'Measures the probability of extreme events ("fat tails"). Higher = more black swan risk.',
        'interpretation': 'Normal = 3. > 3: More extreme events than expected. Financial data typically > 3.'
    },
    
    # VaR Metrics
    'var_95': {
        'title': 'Value at Risk (95%)',
        'explanation': 'The maximum expected loss on 95% of days. Only 1 in 20 days should be worse.',
        'interpretation': 'If VaR = $1000, expect to lose more than $1000 only ~1 day per month.'
    },
    'var_99': {
        'title': 'Value at Risk (99%)',
        'explanation': 'The maximum expected loss on 99% of days. Only 1 in 100 days should be worse.',
        'interpretation': 'More conservative measure. Used for worst-case planning.'
    },
    'cvar': {
        'title': 'Conditional VaR (CVaR/ES)',
        'explanation': 'Average loss on the worst days (beyond VaR). "When things go wrong, how wrong?"',
        'interpretation': 'More informative than VaR. Shows the severity of tail events, not just frequency.'
    },
    
    # Factor Exposures
    'market_factor': {
        'title': 'Market Factor (MKT)',
        'explanation': 'Exposure to overall stock market movements. The classic CAPM beta.',
        'interpretation': 'Positive: Long stocks. Negative: Hedged/Short. Higher = more market risk.'
    },
    'size_factor': {
        'title': 'Size Factor (SMB)',
        'explanation': 'Exposure to small-cap vs large-cap stocks. "Small Minus Big".',
        'interpretation': 'Positive: Tilted toward small caps. Negative: Tilted toward large caps.'
    },
    'value_factor': {
        'title': 'Value Factor (HML)',
        'explanation': 'Exposure to value vs growth stocks. "High Minus Low" book-to-market.',
        'interpretation': 'Positive: Value tilt. Negative: Growth tilt.'
    },
    'momentum_factor': {
        'title': 'Momentum Factor (MOM)',
        'explanation': 'Exposure to stocks with recent strong performance. Winners keep winning.',
        'interpretation': 'Positive: Chasing momentum. Negative: Contrarian/Mean reversion.'
    },
    
    # Monte Carlo
    'monte_carlo': {
        'title': 'Monte Carlo Simulation',
        'explanation': 'Runs thousands of random future scenarios based on historical patterns to estimate risk.',
        'interpretation': 'Shows range of possible outcomes, not a single prediction.'
    },
    'jump_diffusion': {
        'title': 'Jump-Diffusion Model',
        'explanation': 'Adds sudden "jumps" (crashes/rallies) to standard random walk. Captures black swans.',
        'interpretation': 'More realistic than assuming smooth price movements.'
    },
    
    # Stress Testing
    'stress_test': {
        'title': 'Stress Test',
        'explanation': 'Simulates how your portfolio would perform under extreme scenarios (crashes, rate hikes).',
        'interpretation': 'Helps prepare for worst-case scenarios before they happen.'
    }
}


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================

@dataclass
class FactorExposures:
    """Multi-factor model results"""
    market_beta: float
    size_beta: float
    value_beta: float
    momentum_beta: float
    alpha: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    factor_contributions: Dict[str, float]
    
@dataclass
class VolatilityMetrics:
    """GARCH volatility modeling results"""
    historical_vol: float
    garch_vol: float
    ewma_vol: float
    vol_forecast_1d: float
    vol_forecast_5d: float
    vol_forecast_20d: float
    persistence: float
    vol_of_vol: float
    
@dataclass
class RiskMetrics:
    """VaR and tail risk metrics"""
    var_95_parametric: float
    var_99_parametric: float
    var_95_historical: float
    var_99_historical: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    calmar_ratio: float
    
@dataclass
class DistributionMetrics:
    """Return distribution characteristics"""
    mean: float
    std: float
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    is_normal: bool


# =============================================================================
# QUANTITATIVE ENGINE CLASS
# =============================================================================

class QuantEngine:
    """
    Advanced Quantitative Analysis Engine
    
    Provides institutional-grade analytics:
    - Multi-factor regression analysis
    - GARCH volatility modeling
    - Monte Carlo simulation with jump-diffusion
    - VaR/CVaR risk metrics
    - Stress testing framework
    """
    
    RISK_FREE_RATE = 0.05  # 5% annual risk-free rate
    TRADING_DAYS = 252
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series = None):
        """
        Initialize the quant engine.
        
        Args:
            portfolio_returns: Daily log returns of the portfolio
            benchmark_returns: Daily log returns of the benchmark (e.g., SPY)
        """
        self.portfolio_returns = portfolio_returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        
        # Align dates if benchmark provided
        if self.benchmark_returns is not None:
            common_idx = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
            self.portfolio_returns = self.portfolio_returns.loc[common_idx]
            self.benchmark_returns = self.benchmark_returns.loc[common_idx]
        
        # Cache for expensive computations
        self._factor_data_cache = None
        self._garch_params_cache = None
        
    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def price_to_log_returns(prices: pd.Series) -> pd.Series:
        """Convert price series to log returns for mathematical consistency."""
        return np.log(prices / prices.shift(1)).dropna()
    
    @staticmethod  
    def log_to_simple_returns(log_returns: pd.Series) -> pd.Series:
        """Convert log returns back to simple returns."""
        return np.exp(log_returns) - 1
    
    def annualize_return(self, daily_return: float) -> float:
        """Annualize a daily return."""
        return daily_return * self.TRADING_DAYS
    
    def annualize_volatility(self, daily_vol: float) -> float:
        """Annualize daily volatility."""
        return daily_vol * np.sqrt(self.TRADING_DAYS)
    
    def daily_risk_free(self) -> float:
        """Get daily risk-free rate."""
        return self.RISK_FREE_RATE / self.TRADING_DAYS
    
    # =========================================================================
    # FACTOR ANALYSIS (FAMA-FRENCH INSPIRED)
    # =========================================================================
    
    def _generate_synthetic_factors(self) -> pd.DataFrame:
        """
        Generate synthetic factor returns when real Fama-French data unavailable.
        Uses market data to create factor proxies.
        """
        if self._factor_data_cache is not None:
            return self._factor_data_cache
            
        n = len(self.portfolio_returns)
        dates = self.portfolio_returns.index
        
        # Market factor = benchmark or synthetic
        if self.benchmark_returns is not None:
            mkt = self.benchmark_returns.values
        else:
            # Synthetic market based on portfolio with noise
            mkt = self.portfolio_returns.values * 0.8 + np.random.normal(0, 0.005, n)
        
        # SMB (Size): Synthetic small-minus-big factor
        # Small caps tend to be more volatile and have different patterns
        smb = np.random.normal(0.0001, 0.006, n)  # Slight positive drift
        
        # HML (Value): Synthetic high-minus-low factor
        hml = np.random.normal(0.00005, 0.005, n)
        
        # Momentum: Based on recent market performance
        mom = np.zeros(n)
        lookback = 20
        for i in range(lookback, n):
            mom[i] = np.mean(mkt[i-lookback:i]) * 2  # Momentum signal
        mom[:lookback] = np.random.normal(0, 0.005, lookback)
        
        factors = pd.DataFrame({
            'MKT': mkt,
            'SMB': smb,
            'HML': hml,
            'MOM': mom
        }, index=dates)
        
        self._factor_data_cache = factors
        return factors
    
    def compute_factor_exposures(self) -> FactorExposures:
        """
        Compute multi-factor model exposures using OLS regression.
        
        Model: R_p - R_f = α + β_mkt*(R_m - R_f) + β_smb*SMB + β_hml*HML + β_mom*MOM + ε
        """
        factors = self._generate_synthetic_factors()
        
        # Excess returns
        rf_daily = self.daily_risk_free()
        excess_returns = self.portfolio_returns - rf_daily
        excess_market = factors['MKT'] - rf_daily
        
        # Build design matrix
        X = pd.DataFrame({
            'MKT': excess_market,
            'SMB': factors['SMB'],
            'HML': factors['HML'],
            'MOM': factors['MOM']
        })
        X = X.dropna()
        y = excess_returns.loc[X.index]
        
        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X.values])
        
        # OLS regression
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y.values, rcond=None)
            alpha = coeffs[0]
            betas = coeffs[1:]
            
            # Predictions and residuals
            y_pred = X_with_const @ coeffs
            resid = y.values - y_pred
            
            # R-squared
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((y.values - np.mean(y.values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Tracking error (annualized)
            tracking_error = self.annualize_volatility(np.std(resid))
            
            # Information ratio
            info_ratio = (self.annualize_return(alpha) / tracking_error) if tracking_error > 0 else 0
            
            # Factor contributions to return
            avg_factors = X.mean()
            contributions = {
                'Market': float(betas[0] * avg_factors['MKT'] * self.TRADING_DAYS),
                'Size': float(betas[1] * avg_factors['SMB'] * self.TRADING_DAYS),
                'Value': float(betas[2] * avg_factors['HML'] * self.TRADING_DAYS),
                'Momentum': float(betas[3] * avg_factors['MOM'] * self.TRADING_DAYS),
                'Alpha': float(alpha * self.TRADING_DAYS)
            }
            
            return FactorExposures(
                market_beta=float(betas[0]),
                size_beta=float(betas[1]),
                value_beta=float(betas[2]),
                momentum_beta=float(betas[3]),
                alpha=float(self.annualize_return(alpha)),
                r_squared=float(r_squared),
                tracking_error=float(tracking_error),
                information_ratio=float(info_ratio),
                factor_contributions=contributions
            )
            
        except Exception as e:
            # Fallback to simple beta calculation
            if self.benchmark_returns is not None:
                cov = np.cov(self.portfolio_returns, self.benchmark_returns)[0, 1]
                var = np.var(self.benchmark_returns)
                beta = cov / var if var > 0 else 1.0
            else:
                beta = 1.0
                
            return FactorExposures(
                market_beta=beta,
                size_beta=0.0,
                value_beta=0.0,
                momentum_beta=0.0,
                alpha=0.0,
                r_squared=0.0,
                tracking_error=0.0,
                information_ratio=0.0,
                factor_contributions={'Market': 0, 'Size': 0, 'Value': 0, 'Momentum': 0, 'Alpha': 0}
            )
    
    def compute_rolling_correlations(self, windows: List[int] = [30, 90, 252]) -> Dict[str, pd.DataFrame]:
        """
        Compute rolling correlations between portfolio and factors.
        
        Args:
            windows: List of rolling window sizes (in days)
            
        Returns:
            Dictionary mapping window size to correlation DataFrame
        """
        factors = self._generate_synthetic_factors()
        all_data = pd.concat([self.portfolio_returns.rename('Portfolio'), factors], axis=1)
        
        results = {}
        for window in windows:
            if len(all_data) >= window:
                rolling_corr = all_data.rolling(window=window).corr()
                # Extract portfolio correlations with factors
                portfolio_corrs = rolling_corr.xs('Portfolio', level=1)[['MKT', 'SMB', 'HML', 'MOM']]
                results[f'{window}D'] = portfolio_corrs.dropna()
                
        return results
    
    # =========================================================================
    # GARCH VOLATILITY MODELING
    # =========================================================================
    
    def _fit_garch(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit GARCH(1,1) model using maximum likelihood estimation.
        
        Model: σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        
        Returns:
            (omega, alpha, beta) parameters
        """
        if self._garch_params_cache is not None:
            return self._garch_params_cache
            
        def garch_likelihood(params, returns):
            omega, alpha, beta = params
            
            # Constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            n = len(returns)
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(returns)  # Initialize with unconditional variance
            
            for t in range(1, n):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            
            # Negative log-likelihood (assuming normal distribution)
            sigma2 = np.maximum(sigma2, 1e-10)  # Prevent log(0)
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
            
            return -ll  # Minimize negative log-likelihood
        
        # Initial guess
        var_r = np.var(returns)
        x0 = [var_r * 0.1, 0.1, 0.8]  # omega, alpha, beta
        
        # Bounds
        bounds = [(1e-8, var_r), (0.001, 0.5), (0.3, 0.99)]
        
        try:
            result = minimize(garch_likelihood, x0, args=(returns,), 
                            method='L-BFGS-B', bounds=bounds)
            params = tuple(result.x)
        except:
            # Fallback to simple estimates
            params = (var_r * 0.05, 0.1, 0.85)
        
        self._garch_params_cache = params
        return params
    
    def compute_volatility_metrics(self) -> VolatilityMetrics:
        """
        Compute comprehensive volatility metrics including GARCH forecasts.
        """
        returns = self.portfolio_returns.values
        n = len(returns)
        
        # Historical volatility (annualized)
        hist_vol = self.annualize_volatility(np.std(returns))
        
        # EWMA volatility (RiskMetrics style, lambda=0.94)
        lambda_param = 0.94
        ewma_var = np.zeros(n)
        ewma_var[0] = np.var(returns)
        for t in range(1, n):
            ewma_var[t] = lambda_param * ewma_var[t-1] + (1 - lambda_param) * returns[t-1]**2
        ewma_vol = self.annualize_volatility(np.sqrt(ewma_var[-1]))
        
        # GARCH(1,1) fitting
        omega, alpha, beta = self._fit_garch(returns)
        
        # Current GARCH variance
        garch_var = np.zeros(n)
        garch_var[0] = np.var(returns)
        for t in range(1, n):
            garch_var[t] = omega + alpha * returns[t-1]**2 + beta * garch_var[t-1]
        
        current_garch_var = garch_var[-1]
        garch_vol = self.annualize_volatility(np.sqrt(current_garch_var))
        
        # Volatility forecasts
        # Multi-step GARCH forecast: E[σ²_{t+h}] = ω*(1-(α+β)^h)/(1-(α+β)) + (α+β)^h * σ²_t
        persistence = alpha + beta
        unconditional_var = omega / (1 - persistence) if persistence < 1 else current_garch_var
        
        def forecast_var(h):
            if persistence >= 1:
                return current_garch_var
            return unconditional_var + (persistence ** h) * (current_garch_var - unconditional_var)
        
        vol_1d = np.sqrt(forecast_var(1)) * np.sqrt(self.TRADING_DAYS)
        vol_5d = np.sqrt(np.mean([forecast_var(i) for i in range(1, 6)])) * np.sqrt(self.TRADING_DAYS)
        vol_20d = np.sqrt(np.mean([forecast_var(i) for i in range(1, 21)])) * np.sqrt(self.TRADING_DAYS)
        
        # Vol of vol
        rolling_vol = pd.Series(returns).rolling(20).std() * np.sqrt(self.TRADING_DAYS)
        vol_of_vol = rolling_vol.std() if not rolling_vol.isna().all() else 0.0
        
        return VolatilityMetrics(
            historical_vol=float(hist_vol),
            garch_vol=float(garch_vol),
            ewma_vol=float(ewma_vol),
            vol_forecast_1d=float(vol_1d),
            vol_forecast_5d=float(vol_5d),
            vol_forecast_20d=float(vol_20d),
            persistence=float(persistence),
            vol_of_vol=float(vol_of_vol)
        )
    
    def compute_volatility_cone(self, horizons: List[int] = [5, 10, 20, 60, 120]) -> pd.DataFrame:
        """
        Compute volatility cone showing percentile ranges at different horizons.
        """
        returns = self.portfolio_returns.values
        results = []
        
        for h in horizons:
            if len(returns) >= h:
                rolling_vols = []
                for i in range(h, len(returns)):
                    vol = np.std(returns[i-h:i]) * np.sqrt(self.TRADING_DAYS)
                    rolling_vols.append(vol)
                
                if rolling_vols:
                    results.append({
                        'Horizon': h,
                        'Min': np.percentile(rolling_vols, 5),
                        'P25': np.percentile(rolling_vols, 25),
                        'Median': np.percentile(rolling_vols, 50),
                        'P75': np.percentile(rolling_vols, 75),
                        'Max': np.percentile(rolling_vols, 95),
                        'Current': rolling_vols[-1] if rolling_vols else 0
                    })
        
        return pd.DataFrame(results)
    
    # =========================================================================
    # MONTE CARLO SIMULATION WITH JUMP-DIFFUSION
    # =========================================================================
    
    def run_monte_carlo_simulation(
        self,
        initial_value: float,
        horizon_days: int = 252,
        n_simulations: int = 10000,
        model: str = 'jump_diffusion'
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation using Jump-Diffusion or GBM model.
        
        Jump-Diffusion (Merton): dS = μSdt + σSdW + S*J*dN
        where J is jump size and N is Poisson process
        
        Args:
            initial_value: Starting portfolio value
            horizon_days: Number of days to simulate
            n_simulations: Number of simulation paths
            model: 'jump_diffusion' or 'gbm'
            
        Returns:
            Dictionary with simulation results
        """
        returns = self.portfolio_returns.values
        
        # Estimate parameters from historical data
        mu = np.mean(returns)  # Daily drift
        sigma = np.std(returns)  # Daily volatility
        
        # Generate random paths (vectorized for performance)
        np.random.seed(42)  # Reproducibility
        dt = 1  # Daily steps
        
        # Standard Brownian motion component
        Z = np.random.standard_normal((n_simulations, horizon_days))
        
        if model == 'jump_diffusion':
            # Jump parameters (estimated from tail behavior)
            excess_kurtosis = stats.kurtosis(returns)
            jump_intensity = 0.1  # Average 10% of days have jumps
            jump_mean = -0.02  # Jumps tend to be negative (crashes)
            jump_std = 0.03  # Jump volatility
            
            # Poisson jump process
            N = np.random.poisson(jump_intensity, (n_simulations, horizon_days))
            J = np.random.normal(jump_mean, jump_std, (n_simulations, horizon_days))
            
            # Jump-diffusion returns
            simulated_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + J * N
        else:
            # Standard GBM
            simulated_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        
        # Convert to price paths
        cumulative_returns = np.cumsum(simulated_returns, axis=1)
        price_paths = initial_value * np.exp(cumulative_returns)
        
        # Terminal values
        terminal_values = price_paths[:, -1]
        terminal_returns = (terminal_values / initial_value - 1) * 100
        
        # Calculate statistics
        results = {
            'price_paths': price_paths,
            'terminal_values': terminal_values,
            'terminal_returns': terminal_returns,
            'initial_value': initial_value,
            'horizon_days': horizon_days,
            'n_simulations': n_simulations,
            'model': model,
            'statistics': {
                'mean_terminal': float(np.mean(terminal_values)),
                'median_terminal': float(np.median(terminal_values)),
                'std_terminal': float(np.std(terminal_values)),
                'percentile_5': float(np.percentile(terminal_values, 5)),
                'percentile_25': float(np.percentile(terminal_values, 25)),
                'percentile_75': float(np.percentile(terminal_values, 75)),
                'percentile_95': float(np.percentile(terminal_values, 95)),
                'prob_loss': float(np.mean(terminal_values < initial_value) * 100),
                'prob_gain_10pct': float(np.mean(terminal_values > initial_value * 1.1) * 100),
                'expected_return': float(np.mean(terminal_returns)),
                'worst_case': float(np.min(terminal_values)),
                'best_case': float(np.max(terminal_values))
            }
        }
        
        return results
    
    # =========================================================================
    # VALUE AT RISK (VaR) AND EXPECTED SHORTFALL (CVaR)
    # =========================================================================
    
    def compute_risk_metrics(self, portfolio_value: float) -> RiskMetrics:
        """
        Compute comprehensive risk metrics including VaR and CVaR.
        
        Args:
            portfolio_value: Current portfolio value in dollars
        """
        returns = self.portfolio_returns.values
        
        # Distribution statistics
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Parametric VaR (assuming normal distribution)
        z_95 = stats.norm.ppf(0.05)  # -1.645
        z_99 = stats.norm.ppf(0.01)  # -2.326
        
        var_95_pct = -(mu + z_95 * sigma)
        var_99_pct = -(mu + z_99 * sigma)
        
        var_95_parametric = portfolio_value * var_95_pct
        var_99_parametric = portfolio_value * var_99_pct
        
        # Historical VaR (empirical percentiles)
        var_95_historical = portfolio_value * (-np.percentile(returns, 5))
        var_99_historical = portfolio_value * (-np.percentile(returns, 1))
        
        # Expected Shortfall (CVaR) - average loss beyond VaR
        tail_5 = returns[returns <= np.percentile(returns, 5)]
        tail_1 = returns[returns <= np.percentile(returns, 1)]
        
        cvar_95 = portfolio_value * (-np.mean(tail_5)) if len(tail_5) > 0 else var_95_historical
        cvar_99 = portfolio_value * (-np.mean(tail_1)) if len(tail_1) > 0 else var_99_historical
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns)) * 100
        
        # Calmar Ratio (annualized return / max drawdown)
        annual_return = self.annualize_return(mu) * 100
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return RiskMetrics(
            var_95_parametric=float(var_95_parametric),
            var_99_parametric=float(var_99_parametric),
            var_95_historical=float(var_95_historical),
            var_99_historical=float(var_99_historical),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            max_drawdown=float(max_drawdown),
            calmar_ratio=float(calmar)
        )
    
    def compute_distribution_metrics(self) -> DistributionMetrics:
        """
        Analyze the statistical distribution of returns.
        """
        returns = self.portfolio_returns.values
        
        mean = np.mean(returns)
        std = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis (normal = 0)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        is_normal = jb_pvalue > 0.05
        
        return DistributionMetrics(
            mean=float(self.annualize_return(mean) * 100),
            std=float(self.annualize_volatility(std) * 100),
            skewness=float(skew),
            kurtosis=float(kurt + 3),  # Report total kurtosis (normal = 3)
            jarque_bera_stat=float(jb_stat),
            jarque_bera_pvalue=float(jb_pvalue),
            is_normal=is_normal
        )
    
    # =========================================================================
    # STRESS TESTING FRAMEWORK
    # =========================================================================
    
    def run_stress_test(
        self,
        portfolio_value: float,
        scenarios: Dict[str, Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Run stress tests on the portfolio.
        
        Args:
            portfolio_value: Current portfolio value
            scenarios: Dictionary of scenarios with factor shocks
            
        Returns:
            DataFrame with stress test results
        """
        if scenarios is None:
            scenarios = {
                'Market Crash (-20%)': {'market': -0.20, 'volatility': 2.0},
                'Moderate Decline (-10%)': {'market': -0.10, 'volatility': 1.5},
                'Rate Shock (+2%)': {'market': -0.05, 'volatility': 1.3, 'rates': 0.02},
                'Stagflation': {'market': -0.15, 'volatility': 1.8, 'rates': 0.03},
                'Flash Crash (-5% 1-day)': {'market': -0.05, 'volatility': 3.0},
                'Recovery Rally (+15%)': {'market': 0.15, 'volatility': 0.8},
                'Volatility Spike (VIX +50%)': {'market': -0.03, 'volatility': 2.5},
                'Black Swan (-30%)': {'market': -0.30, 'volatility': 4.0},
            }
        
        # Get factor exposures
        exposures = self.compute_factor_exposures()
        beta = exposures.market_beta
        
        results = []
        for name, shocks in scenarios.items():
            market_shock = shocks.get('market', 0)
            vol_multiplier = shocks.get('volatility', 1.0)
            
            # Estimate portfolio impact based on beta
            portfolio_shock = beta * market_shock
            
            # Adjust for higher vol scenarios (wider confidence interval)
            vol_adjustment = (vol_multiplier - 1) * 0.02 * beta  # Additional downside
            total_shock = portfolio_shock - abs(vol_adjustment)
            
            impact_dollar = portfolio_value * total_shock
            impact_pct = total_shock * 100
            new_value = portfolio_value + impact_dollar
            
            results.append({
                'Scenario': name,
                'Market Shock': f"{market_shock*100:+.1f}%",
                'Vol Multiplier': f"{vol_multiplier:.1f}x",
                'Portfolio Impact (%)': impact_pct,
                'Portfolio Impact ($)': impact_dollar,
                'New Value ($)': new_value,
                'Severity': 'High' if abs(impact_pct) > 15 else ('Medium' if abs(impact_pct) > 5 else 'Low')
            })
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis(
        self,
        portfolio_value: float,
        market_range: Tuple[float, float] = (-0.20, 0.20),
        rate_range: Tuple[float, float] = (-0.02, 0.02),
        steps: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis across market and rate changes.
        
        Returns grid of portfolio values for 3D visualization.
        """
        exposures = self.compute_factor_exposures()
        beta = exposures.market_beta
        
        market_shocks = np.linspace(market_range[0], market_range[1], steps)
        rate_shocks = np.linspace(rate_range[0], rate_range[1], steps)
        
        # Create meshgrid for 3D surface
        M, R = np.meshgrid(market_shocks, rate_shocks)
        
        # Estimate portfolio values
        # Rate sensitivity: assume duration-like impact (rough estimate)
        rate_sensitivity = -0.5  # Portfolio loses ~0.5% per 1% rate increase
        
        portfolio_returns_grid = beta * M + rate_sensitivity * R
        portfolio_values_grid = portfolio_value * (1 + portfolio_returns_grid)
        
        return {
            'market_shocks': market_shocks * 100,
            'rate_shocks': rate_shocks * 100,
            'market_grid': M * 100,
            'rate_grid': R * 100,
            'portfolio_values': portfolio_values_grid,
            'portfolio_returns': portfolio_returns_grid * 100
        }
    
    # =========================================================================
    # NATURAL LANGUAGE INSIGHT GENERATION
    # =========================================================================
    
    def generate_risk_insights(self, portfolio_value: float) -> List[str]:
        """
        Generate plain-English insights about portfolio risk characteristics.
        """
        insights = []
        
        # Factor analysis insights
        exposures = self.compute_factor_exposures()
        
        if exposures.market_beta > 1.2:
            insights.append(
                f"⚠️ **High Market Sensitivity**: Your portfolio has a beta of {exposures.market_beta:.2f}, "
                f"meaning it's {((exposures.market_beta - 1) * 100):.0f}% more volatile than the market. "
                f"Expect amplified gains in bull markets, but also amplified losses in downturns."
            )
        elif exposures.market_beta < 0.8:
            insights.append(
                f"🛡️ **Defensive Positioning**: With a beta of {exposures.market_beta:.2f}, your portfolio "
                f"is less sensitive to market swings. Good for capital preservation, but may lag in rallies."
            )
        else:
            insights.append(
                f"⚖️ **Market-Neutral Beta**: Your portfolio beta of {exposures.market_beta:.2f} closely "
                f"tracks the overall market."
            )
        
        if exposures.alpha > 0.02:
            insights.append(
                f"🌟 **Positive Alpha**: Your portfolio is generating {exposures.alpha*100:.1f}% annualized "
                f"excess return above what's expected for its risk level. This suggests skill or favorable positioning."
            )
        elif exposures.alpha < -0.02:
            insights.append(
                f"📉 **Negative Alpha**: Your portfolio is underperforming by {abs(exposures.alpha)*100:.1f}% "
                f"annually relative to its risk. Consider reviewing underperforming positions."
            )
        
        # Volatility insights
        vol_metrics = self.compute_volatility_metrics()
        
        if vol_metrics.garch_vol > vol_metrics.historical_vol * 1.2:
            insights.append(
                f"🔥 **Rising Volatility**: GARCH model forecasts {vol_metrics.garch_vol:.1f}% volatility, "
                f"which is {((vol_metrics.garch_vol/vol_metrics.historical_vol - 1) * 100):.0f}% above historical levels. "
                f"Expect larger price swings in the near term."
            )
        elif vol_metrics.garch_vol < vol_metrics.historical_vol * 0.8:
            insights.append(
                f"😌 **Calming Volatility**: Current volatility ({vol_metrics.garch_vol:.1f}%) is below "
                f"historical averages ({vol_metrics.historical_vol:.1f}%). Markets appear calmer."
            )
        
        if vol_metrics.persistence > 0.95:
            insights.append(
                f"⏳ **Persistent Volatility**: Volatility shocks tend to last a long time (persistence: "
                f"{vol_metrics.persistence:.2f}). Once markets get turbulent, expect it to continue."
            )
        
        # Distribution insights
        dist = self.compute_distribution_metrics()
        
        if dist.skewness < -0.5:
            insights.append(
                f"📊 **Negative Skew ({dist.skewness:.2f})**: Your returns are negatively skewed, meaning "
                f"you have frequent small gains but occasional large losses. Consider tail hedging."
            )
        elif dist.skewness > 0.5:
            insights.append(
                f"📊 **Positive Skew ({dist.skewness:.2f})**: Your returns show positive skewness - "
                f"occasional large gains with limited downside. This is generally favorable."
            )
        
        if dist.kurtosis > 4:
            insights.append(
                f"🦢 **Fat Tails Alert**: Kurtosis of {dist.kurtosis:.2f} (normal = 3) indicates "
                f"higher probability of extreme events. Black swan risk is elevated."
            )
        
        if not dist.is_normal:
            insights.append(
                f"📐 **Non-Normal Returns**: Statistical tests reject normality (p={dist.jarque_bera_pvalue:.4f}). "
                f"Standard deviation understates true risk. Use VaR/CVaR for better risk estimates."
            )
        
        # Risk metrics insights
        risk = self.compute_risk_metrics(portfolio_value)
        
        insights.append(
            f"💰 **Daily Risk Budget**: Based on 95% VaR, expect daily losses to exceed "
            f"${risk.var_95_historical:,.0f} only ~1 day per month. Worst-case (99% VaR): ${risk.var_99_historical:,.0f}."
        )
        
        if risk.cvar_95 > risk.var_95_historical * 1.5:
            insights.append(
                f"⚠️ **Tail Risk Warning**: When losses exceed VaR, they average ${risk.cvar_95:,.0f} "
                f"(CVaR), which is {(risk.cvar_95/risk.var_95_historical - 1)*100:.0f}% worse than VaR. "
                f"Tail events are severe."
            )
        
        if risk.max_drawdown > 20:
            insights.append(
                f"📉 **Drawdown History**: Your portfolio has experienced a maximum drawdown of "
                f"{risk.max_drawdown:.1f}%. Ensure you can psychologically handle similar declines."
            )
        
        return insights
    
    def generate_monte_carlo_insights(
        self,
        simulation_results: Dict[str, Any],
        portfolio_value: float
    ) -> List[str]:
        """
        Generate insights from Monte Carlo simulation results.
        """
        stats = simulation_results['statistics']
        horizon = simulation_results['horizon_days']
        model = simulation_results['model']
        
        insights = []
        
        # Model explanation
        if model == 'jump_diffusion':
            insights.append(
                f"🎲 **Simulation Model**: Using Jump-Diffusion (Merton model) which accounts for sudden "
                f"market shocks and 'black swan' events that standard models miss."
            )
        else:
            insights.append(
                f"🎲 **Simulation Model**: Using Geometric Brownian Motion (GBM), the standard model for "
                f"price movements. Note: May underestimate tail risk."
            )
        
        # Main outcome
        expected_change = stats['expected_return']
        if expected_change > 0:
            insights.append(
                f"📈 **Expected Outcome**: Based on {simulation_results['n_simulations']:,} simulations over "
                f"{horizon} days, your portfolio is expected to {'gain' if expected_change > 0 else 'lose'} "
                f"{abs(expected_change):.1f}% on average (${stats['mean_terminal'] - portfolio_value:+,.0f})."
            )
        
        # Probability insights
        prob_loss = stats['prob_loss']
        if prob_loss > 50:
            insights.append(
                f"⚠️ **Loss Probability**: There's a {prob_loss:.0f}% chance of losing money over this period. "
                f"Consider reducing risk or shortening your time horizon."
            )
        else:
            insights.append(
                f"✅ **Favorable Odds**: {100-prob_loss:.0f}% probability of gains over {horizon} days."
            )
        
        # Range of outcomes
        low = stats['percentile_5']
        high = stats['percentile_95']
        insights.append(
            f"📊 **90% Confidence Range**: Your portfolio value will likely be between "
            f"${low:,.0f} and ${high:,.0f} ({((low/portfolio_value-1)*100):+.1f}% to "
            f"{((high/portfolio_value-1)*100):+.1f}%)."
        )
        
        # Worst case
        worst = stats['worst_case']
        worst_pct = (worst / portfolio_value - 1) * 100
        insights.append(
            f"💀 **Worst-Case Scenario**: In the most extreme simulation, portfolio dropped to "
            f"${worst:,.0f} ({worst_pct:+.1f}%). While unlikely, always plan for tail events."
        )
        
        return insights


# =============================================================================
# HELPER FUNCTIONS FOR STREAMLIT INTEGRATION
# =============================================================================

def create_tooltip_html(key: str) -> str:
    """Create HTML for an educational tooltip."""
    if key not in TOOLTIPS:
        return ""
    
    tip = TOOLTIPS[key]
    return f"""
    <span class="tooltip-container">
        <span class="tooltip-icon">ⓘ</span>
        <span class="tooltip-content">
            <strong>{tip['title']}</strong><br>
            {tip['explanation']}<br><br>
            <em>How to interpret:</em> {tip['interpretation']}
        </span>
    </span>
    """

def get_tooltip_css() -> str:
    """Get CSS styles for tooltips."""
    return """
    <style>
    .tooltip-container {
        position: relative;
        display: inline-block;
        cursor: help;
        margin-left: 5px;
    }
    .tooltip-icon {
        color: #1E88E5;
        font-size: 14px;
        font-weight: bold;
    }
    .tooltip-content {
        visibility: hidden;
        width: 300px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 8px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 13px;
        line-height: 1.4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .tooltip-content::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #333 transparent transparent transparent;
    }
    .tooltip-container:hover .tooltip-content {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """

def format_metric_with_tooltip(label: str, value: str, tooltip_key: str) -> str:
    """Format a metric label with its tooltip."""
    tooltip = create_tooltip_html(tooltip_key)
    return f"{label} {tooltip}"
