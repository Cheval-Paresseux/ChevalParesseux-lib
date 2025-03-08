import numpy as np
import pandas as pd


def compute_stats(
    returns: pd.Series,
    market_returns: pd.Series = None,
    risk_free_rate: float = 0.0,
    frequence: str = "daily",
):
    """
    Compute the statistics of the investment.

    Args:
        returns (pd.Series): Series of returns of the investment.
        market_returns (pd.Series): Series of returns of the market index for comparison.
        risk_free_rate (float): Risk-free rate for certain calculations.
        frequence (str): Frequence of the returns.

    Returns:
        stats (dict): Dictionary containing the statistics of the investment, including:

        ======= Returns distribution statistics =======
        - **Expected Return**: The annualized mean return, indicating average performance.
        - **Volatility**: Standard deviation of returns, representing total risk.
        - **Downside Deviation**: Standard deviation of negative returns, used in risk-adjusted metrics like Sortino Ratio.
        - **Median Return**: The median of returns, a measure of central tendency.
        - **Skew** and **Kurtosis**: Describe the distribution shape, with skew indicating asymmetry and kurtosis indicating tail heaviness.

        ======= Risk measures =======
        - **Maximum Drawdown**: Largest observed loss from peak to trough, a measure of downside risk.
        - **Max Drawdown Duration**: Longest period to recover from drawdown, indicating risk recovery time.
        - **VaR 95** and **CVaR 95**: Value at Risk and Conditional Value at Risk at 95%, giving the maximum and average expected losses in worst-case scenarios.

        ======= Market sensitivity measures =======
        - **Beta**: Sensitivity to market movements.
        - **Alpha**: Risk-adjusted return above the market return.
        - **Upside/Downside Capture Ratios**: Percent of market gains or losses captured by the investment.
        - **Tracking Error**: Volatility of return differences from the market.

        ======= Performance measures =======
        - **Sharpe**: Risk-adjusted returns per unit of volatility.
        - **Sortino Ratio**: Risk-adjusted return accounting only for downside volatility.
        - **Treynor Ratio**: Return per unit of systematic (market) risk.
        - **Information Ratio**: Excess return per unit of tracking error.

        - **Sterling Ratio**: Return per unit of average drawdown.
        - **Calmar Ratio**: Return per unit of maximum drawdown.
    """
    # ======= 0. Initialization =======
    frequence_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequence = frequence_dict[frequence]

    cumulative_returns = (1 + returns).cumprod()

    # ======= I. Returns distribution statistics =======
    expected_return = returns.mean() * adjusted_frequence
    volatility = returns.std() * np.sqrt(adjusted_frequence)
    downside_deviation = returns[returns < 0].std() * np.sqrt(adjusted_frequence) if returns[returns < 0].sum() != 0 else 0
    median_return = returns.median() * adjusted_frequence
    skew = returns.skew()
    kurtosis = returns.kurtosis()

    # ======= II. Risk measures =======
    # ------ Maximum Drawdown and Duration
    running_max = cumulative_returns.cummax().replace(0, 1e-10)
    drawdown = (cumulative_returns / running_max) - 1
    drawdown_durations = (drawdown < 0).astype(int).groupby((drawdown == 0).cumsum()).cumsum()

    maximum_drawdown = drawdown.min()
    max_drawdown_duration = drawdown_durations.max()

    # ------ Value at Risk and Conditional Value at Risk
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()

    # ======= III. Market sensitivity measures =======
    if market_returns is None:
        beta = 0
        alpha = 0
        upside_capture = 0
        downside_capture = 0
        tracking_error = 0
    else:
        # ------ Beta and Alpha (Jensens's)
        beta = returns.cov(market_returns) / market_returns.var()
        alpha = expected_return - beta * (market_returns.mean() * adjusted_frequence)

        # ------ Capture Ratios
        upside_capture = returns[market_returns > 0].mean() / market_returns[market_returns > 0].mean()
        downside_capture = returns[market_returns < 0].mean() / market_returns[market_returns < 0].mean()

        # ------ Tracking Error
        tracking_error = returns.sub(market_returns).std() * np.sqrt(adjusted_frequence)

    # ======= IV. Performance measures =======
    # ------ Sharpe, Sortino, Treynor, and Information Ratios
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility != 0 else 0
    sortino_ratio = expected_return / downside_deviation if downside_deviation != 0 else 0
    treynor_ratio = expected_return / beta if beta != 0 else 0
    information_ratio = (expected_return - market_returns.mean() * adjusted_frequence) / tracking_error if tracking_error != 0 else 0

    # ------ Sterling, and Calmar Ratios
    average_drawdown = abs(drawdown[drawdown < 0].mean()) if drawdown[drawdown < 0].sum() != 0 else 0
    sterling_ratio = (expected_return - risk_free_rate) / average_drawdown if average_drawdown != 0 else 0
    calmar_ratio = expected_return / abs(maximum_drawdown) if maximum_drawdown != 0 else 0

    # ======= IV. Store the statistics =======
    stats = {
        "expected_return": expected_return,
        "volatility": volatility,
        "downside_deviation": downside_deviation,
        "median_return": median_return,
        "skew": skew,
        "kurtosis": kurtosis,
        "maximum_drawdown": maximum_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "beta": beta,
        "alpha": alpha,
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
        "tracking_error": tracking_error,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "treynor_ratio": treynor_ratio,
        "information_ratio": information_ratio,
        "sterling_ratio": sterling_ratio,
        "calmar_ratio": calmar_ratio,
    }

    return stats
