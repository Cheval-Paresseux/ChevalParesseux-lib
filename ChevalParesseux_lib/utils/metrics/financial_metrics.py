import numpy as np
import pandas as pd



#! ==================================================================================== #
#! ============================== General Functions =================================== #




#! ==================================================================================== #
#! ================================ Financial Metrics  ================================ #
def get_distribution(
    returns_series: pd.Series, 
    frequency: str = "daily"
) -> dict:
    """
    Compute distribution statistics for a return series at a given frequency.

    Parameters:
        - returns_series (pd.Series): Series of returns (e.g., daily or intraday).
        - frequency (str): Frequency of the return series. Must be one of:
                           'daily', '5m', or '1m'.
                           Default is 'daily'.

    Returns:
        - distribution_stats (dict): Dictionary containing the following statistics:
            - expected_return: Annualized mean return.
            - volatility: Annualized standard deviation of returns.
            - downside_deviation: Annualized standard deviation of negative returns.
            - median_return: Annualized median return.
            - skew: Skewness of the return distribution.
            - kurtosis: Kurtosis of the return distribution.
    """
    # ======= I. Get the right frequency =======
    frequency_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequency = frequency_dict[frequency]
    
    # ======= II. Compute the statistics =======
    expected_return = returns_series.mean() * adjusted_frequency
    volatility = returns_series.std() * np.sqrt(adjusted_frequency)
    downside_deviation = returns_series[returns_series < 0].std() * np.sqrt(adjusted_frequency) if returns_series[returns_series < 0].sum() != 0 else 0
    median_return = returns_series.median() * adjusted_frequency
    skew = returns_series.skew()
    kurtosis = returns_series.kurtosis()
    
    # ======= III. Store the statistics =======
    distribution_stats = {
        "expected_return": expected_return,
        "volatility": volatility,
        "downside_deviation": downside_deviation,
        "median_return": median_return,
        "skew": skew,
        "kurtosis": kurtosis,
    }
    
    return distribution_stats

#*____________________________________________________________________________________ #
def get_risk_measures(
    returns_series: pd.Series
) -> dict:
    """
    Compute key downside risk metrics for a return series.

    Parameters:
        - returns_series (pd.Series): Series of periodic returns.

    Returns:
        - risk_stats (dict): Dictionary containing:
            - mean_drawdown: Average drawdown over the period.
            - maximum_drawdown: Maximum observed drawdown.
            - max_drawdown_duration: Maximum duration of drawdowns (in periods).
            - var_95: 5% Value at Risk (VaR).
            - cvar_95: Conditional VaR at 5% (Expected Shortfall).
    """
    # ======= I. Compute the Cumulative returns =======    
    cumulative_returns = (1 + returns_series).cumprod()
    
    # ======= II. Compute the statistics =======
    # ------ Maximum Drawdown and Duration
    running_max = cumulative_returns.cummax().replace(0, 1e-10)
    drawdown = (cumulative_returns / running_max) - 1
    drawdown_durations = (drawdown < 0).astype(int).groupby((drawdown == 0).cumsum()).cumsum()
    
    mean_drawdown = drawdown.mean()

    maximum_drawdown = drawdown.min()
    max_drawdown_duration = drawdown_durations.max()

    # ------ Value at Risk and Conditional Value at Risk
    var_95 = returns_series.quantile(0.05)
    cvar_95 = returns_series[returns_series <= var_95].mean()
    
    # ======= III. Store the statistics =======
    risk_stats = {
        "mean_drawdown": mean_drawdown,
        "maximum_drawdown": maximum_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }
    
    return risk_stats

#*____________________________________________________________________________________ #
def get_market_sensitivity(
    returns_series: pd.Series, 
    market_returns: pd.Series, 
    frequency: str = "daily"
) -> dict:
    """
    Estimate the sensitivity of a strategy or asset to market returns.

    Parameters:
        - returns_series (pd.Series): Asset or strategy return series.
        - market_returns (pd.Series): Benchmark or market return series.
        - frequency (str): Frequency of data ('daily', '5m', or '1m'). Default is 'daily'.

    Returns:
        - market_sensitivity_stats (dict): Dictionary containing:
            - beta: Market beta coefficient.
            - alpha: Jensen's alpha (annualized).
            - upside_capture: Average return ratio in positive market periods.
            - downside_capture: Average return ratio in negative market periods.
            - tracking_error: Annualized tracking error vs. the market.
    """
    # ======= I. Get the right frequency =======
    frequency_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequency = frequency_dict[frequency]
    
    # ======= II. Compute the statistics =======
    # ------ Beta and Alpha (Jensens's)
    beta = returns_series.cov(market_returns) / market_returns.var()
    alpha = returns_series.mean() * adjusted_frequency - beta * (market_returns.mean() * adjusted_frequency)
    
    # ------ Capture Ratios
    upside_capture = returns_series[market_returns > 0].mean() / market_returns[market_returns > 0].mean()
    downside_capture = returns_series[market_returns < 0].mean() / market_returns[market_returns < 0].mean()

    # ------ Tracking Error
    tracking_error = returns_series.sub(market_returns).std() * np.sqrt(adjusted_frequency)
    
    # ======= III. Store the statistics =======
    market_sensitivity_stats = {
        "beta": beta,
        "alpha": alpha,
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
        "tracking_error": tracking_error,
    }
    
    return market_sensitivity_stats

#*____________________________________________________________________________________ #
def get_performance_measures(
    returns_series: pd.Series, 
    market_returns: pd.Series, 
    risk_free_rate: float = 0.0, 
    frequency: str = "daily"
) -> tuple:
    """
    Compute classic performance ratios and supporting metrics for a strategy.

    Parameters:
        - returns_series (pd.Series): Strategy or asset return series.
        - market_returns (pd.Series): Benchmark return series.
        - risk_free_rate (float): Annualized risk-free rate. Default is 0.0.
        - frequence (str): Data frequency ('daily', '5m', or '1m'). Default is 'daily'.

    Returns:
        - performance_stats (dict): Dictionary containing:
            - sharpe_ratio
            - sortino_ratio
            - treynor_ratio
            - information_ratio
            - sterling_ratio
            - calmar_ratio
        - details (tuple): Tuple of three dictionaries:
            (distribution_stats, risk_stats, market_sensitivity_stats)
    """
    # ======= I. Get the right frequency =======
    frequency_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequency = frequency_dict[frequency]
    
    # ======= II. Extract Statistics =======
    distribution_stats = get_distribution(returns_series, frequency)
    expected_return = distribution_stats["expected_return"]
    volatility = distribution_stats["volatility"]
    downside_deviation = distribution_stats["downside_deviation"]
    
    risk_stats = get_risk_measures(returns_series)
    maximum_drawdown = risk_stats["maximum_drawdown"]
    
    market_sensitivity_stats = get_market_sensitivity(returns_series, market_returns, frequency)
    beta = market_sensitivity_stats["beta"]
    tracking_error = market_sensitivity_stats["tracking_error"]
    
    # ======= III. Compute the ratios =======
    # ------ Sharpe, Sortino, Treynor, Information and Calmar Ratios
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility != 0 else 0
    sortino_ratio = expected_return / downside_deviation if downside_deviation != 0 else 0
    treynor_ratio = expected_return / beta if beta != 0 else 0
    information_ratio = (expected_return - market_returns.mean() * adjusted_frequency) / tracking_error if tracking_error != 0 else 0
    calmar_ratio = expected_return / abs(maximum_drawdown) if maximum_drawdown != 0 else 0
    
    # ======= IV. Store the statistics =======
    performance_stats = {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "treynor_ratio": treynor_ratio,
        "information_ratio": information_ratio,
        "calmar_ratio": calmar_ratio,
    }
    
    return performance_stats

