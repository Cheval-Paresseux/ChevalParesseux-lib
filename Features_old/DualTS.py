import os

os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
import warnings

# ========================== Dual TS Features ==========================
def cointegration_features(series_1: pd.Series, series_2: pd.Series, window: int):
    """
    Computes cointegration features between two time series.
    
    Args:
        series_1 (pd.Series): The first time series.
        series_2 (pd.Series): The second time series.
        window (int): The size of the rolling window.
    
    Returns:
        beta_series (pd.Series): The regression coefficient.
        intercept_series (pd.Series): The regression intercept.
        adf_p_values (pd.Series): The p-values of the Augmented Dickey-Fuller test.
        kpss_p_values (pd.Series): The p-values of the Kwiatkowski-Phillips-Schmidt-Shin test.
    """
    # ======== I. Initialize Outputs (Pre-allocate for performance) ========
    num_obs = len(series_1) - window
    beta_values = np.full(num_obs, np.nan)
    intercept_values = np.full(num_obs, np.nan)
    adf_p_values = np.full(num_obs, np.nan)
    kpss_p_values = np.full(num_obs, np.nan)
    residuals_values = np.full(num_obs, np.nan)
    
    # ======== II. Iterate Over Observations ========
    for i in range(num_obs):
        # II.1 Extract Time Windows
        series1_window = series_1.iloc[i : i + window]
        series2_window = series_2.iloc[i : i + window]
        
        # II.2 Perform Cointegration Test
        beta, intercept, adf_results, kpss_results, residuals = cointegration_test(series_1=series1_window, series_2=series2_window)
        
        # II.3 Store Results
        beta_values[i] = beta
        intercept_values[i] = intercept
        adf_p_values[i] = adf_results[1]  # Extract p-value
        kpss_p_values[i] = kpss_results[1]  # Extract p-value
        residuals_values[i] = residuals[-1]
    
    # ======== III. Convert to Series ========
    index = series_1.index[window:]
    
    beta_series = pd.Series(beta_values, index=index)
    intercept_series = pd.Series(intercept_values, index=index)
    adf_p_values_series = pd.Series(adf_p_values, index=index)
    kpss_p_values_series = pd.Series(kpss_p_values, index=index)
    residuals_series = pd.Series(residuals_values, index=index)
    
    return beta_series, intercept_series, adf_p_values_series, kpss_p_values_series, residuals_series


# -----------------------------------------------------------------------------
def ornstein_uhlenbeck_features(series_1: pd.Series, series_2: pd.Series, window: int, residuals_weights: np.array = None):
    """
    Computes Ornstein-Uhlenbeck features between two time series.
    
    Args:
        series_1 (pd.Series): The first time series.
        series_2 (pd.Series): The second time series.
        window (int): The size of the rolling window.
        residuals_weights (np.array): The weights for the residuals.
    
    Returns:
        mu_series (pd.Series): The mean reversion level.
        theta_series (pd.Series): The speed of mean reversion.
        sigma_series (pd.Series): The volatility parameter.
        half_life_series (pd.Series): The half-life of mean reversion.
    """
    # ======== I. Initialize Outputs (Pre-allocate for performance) ========
    num_obs = len(series_1) - window
    mu_values = np.full(num_obs, np.nan)
    theta_values = np.full(num_obs, np.nan)
    sigma_values = np.full(num_obs, np.nan)
    half_life_values = np.full(num_obs, np.nan)
    
    # ======== II. Iterate Over Observations ========
    for i in range(num_obs):
        # II.1 Extract Time Windows
        series1_window = series_1.iloc[i : i + window]
        series2_window = series_2.iloc[i : i + window]
        
        # II.2 Extract residuals from cointegration test
        if residuals_weights is None:
            _, _, _, _, residuals = cointegration_test(series_1=series1_window, series_2=series2_window)
        else: 
            residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
        
        # II.3 Perform Ornstein-Uhlenbeck Estimation
        mu, theta, sigma, half_life = ornstein_uhlenbeck_estimation(series=residuals)
        
        # II.4 Store Results
        mu_values[i] = mu
        theta_values[i] = theta
        sigma_values[i] = sigma
        half_life_values[i] = half_life
    
    # ======== III. Convert to Series ========
    index = series_1.index[window:]
    
    mu_series = pd.Series(mu_values, index=index)
    theta_series = pd.Series(theta_values, index=index)
    sigma_series = pd.Series(sigma_values, index=index)
    half_life_series = pd.Series(half_life_values, index=index)
    
    return mu_series, theta_series, sigma_series, half_life_series


# -----------------------------------------------------------------------------
def kalmanOU_features(series_1: pd.Series, series_2: pd.Series, window: int,  smooth_coefficient: float, residuals_weights: np.array = None):
    """
    Computes Kalman-OU features between two time series.
    
    Args:
        series_1 (pd.Series): The first time series.
        series_2 (pd.Series): The second time series.
        window (int): The size of the rolling window.
        smooth_coefficient (float): A factor controlling the level of process noise.
        residuals_weights (np.array): The weights for the residuals.
    
    Returns:
        state_series (pd.Series): The filtered state estimates.
        variance_series (pd.Series): The estimated variances.
    """
    # ======== I. Initialize Outputs (Pre-allocate for performance) ========
    num_obs = len(series_1) - window
    state_values = np.full(num_obs, np.nan)
    variance_values = np.full(num_obs, np.nan)
    
    # ======== II. Iterate Over Observations ========
    for i in range(num_obs):
        # II.1 Extract Time Windows
        series1_window = series_1.iloc[i : i + window]
        series2_window = series_2.iloc[i : i + window]
        
        # II.2 Extract residuals from cointegration test
        if residuals_weights is None:
            _, _, _, _, residuals = cointegration_test(series_1=series1_window, series_2=series2_window)
        else: 
            residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
        
        # II.3 Perform Ornstein-Uhlenbeck Estimation
        filtered_states, variances = kalmanOU_estimation(series=residuals, smooth_coefficient=smooth_coefficient)
        
        # II.4 Store Results
        state_values[i] = filtered_states[-1]
        variance_values[i] = variances[-1]
    
    # ======== III. Convert to Series ========
    index = series_1.index[window:]
    
    state_series = pd.Series(state_values, index=index)
    variance_series = pd.Series(variance_values, index=index)
    
    return state_series, variance_series


# =========================================================================
# ========================== Auxiliary functions ==========================
def cointegration_test(series_1: pd.Series, series_2: pd.Series):
    """
    Performs a cointegration test between two time series.
    
    Args:
        series_1 (pd.Series): The first time series.
        series_2 (pd.Series): The second time series.
    
    Returns:
        beta (float): The regression coefficient.
        intercept (float): The regression intercept.
        adf_results (tuple): The results of the Augmented Dickey-Fuller test.
        kpss_results (tuple): The results of the Kwiatkowski-Phillips-Schmidt-Shin test.
    """
    # ======== I. Perform a Linear Regression ========
    series_1_reshaped = series_1.values.reshape(-1, 1) 
    model = LinearRegression()
    model.fit(series_1_reshaped, series_2)

    # ======== II. Extract Regression Coefficients ========
    beta = model.coef_[0]
    intercept = model.intercept_

    # ======== III. Compute Residuals ========
    residuals = series_2 - (beta * series_1 + intercept)

    # ======== IV. Perform ADF & KPSS Tests ========
    adf_results = adfuller(residuals)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_results = kpss(residuals, regression="c", nlags="auto")

    return beta, intercept, adf_results, kpss_results, residuals


# -----------------------------------------------------------------------------
def ornstein_uhlenbeck_estimation(series: pd.Series):
    """
    Estimate the Ornstein-Uhlenbeck parameters of a time series.
    
    Args:
        series (pd.Series): The time series.
    
    Returns:
        mu (float): The mean reversion level.
        theta (float): The speed of mean reversion.
        sigma (float): The volatility parameter.
        half_life (float): The half-life of mean reversion.
    """
    # ======== I. Initialize series ========
    series_array = np.array(series)
    differentiated_series = np.diff(series_array)
    mu = np.mean(series)
    
    X = series_array[:-1] - mu  # X_t - mu
    Y = differentiated_series  # X_{t+1} - X_t

    # ======== II. Perform OLS regression ========
    reg = sm.OLS(Y, sm.add_constant(X)).fit()

    # ======== III. Extract Parameters ========
    theta = -reg.params[0]
    if theta > 0:
        residuals = reg.resid
        sigma = np.sqrt(np.var(residuals) * 2 * theta)
        half_life = np.log(2) / theta
    else:
        theta = 0
        sigma = 0
        half_life = 0

    return mu, theta, sigma, half_life


# -----------------------------------------------------------------------------
def kalmanOU_estimation(series: pd.Series, smooth_coefficient: float):
    """
    Applies a Kalman Filter to estimate the hidden state of an Ornstein-Uhlenbeck process.

    Args:
        series (pd.Series): The input time series.
        smooth_coefficient (float): A factor controlling the level of process noise.

    Returns:
        pd.Series: The filtered state estimates.
        pd.Series: The estimated variances.
    """
    # ======== 0. Define Kalman Filter Prediction Step ========
    def make_prediction(observation: float, prior_estimate: float, prior_variance: float,
                        mean: float, theta: float, obs_sigma: float, pro_sigma: float):
        """
        Performs a Kalman Filter update step for the Ornstein-Uhlenbeck process.
        """
        # ======= I. Observation update =======
        innovation_t = observation - prior_estimate
        innovation_variance_t = prior_variance + obs_sigma**2
        kalman_gain_t = prior_variance / innovation_variance_t

        # ======= II. Update state and variance =======
        estimate_t = prior_estimate + kalman_gain_t * innovation_t
        variance_t = (1 - kalman_gain_t) * prior_variance

        # ======= III. Prediction step (OU transition) =======
        estimate_t = mean + (1 - theta) * (estimate_t - mean)
        variance_t = max((1 - theta) ** 2 * variance_t + pro_sigma**2, 1e-8)  # Ensure non-negative variance

        return estimate_t, variance_t

    # ======== I. Estimate OU parameters ========
    mu, theta, sigma, _ = ornstein_uhlenbeck_estimation(series)
    theta = max(theta, 1e-4)

    # ======== II. Initialize Kalman Filter ========
    kf_mean, kf_theta, kf_obs_sigma = mu, theta, sigma
    kf_pro_sigma = kf_obs_sigma * smooth_coefficient  # Process noise scaled by smooth coefficient

    # Initial state estimates
    prior_estimate = kf_mean
    prior_variance = max(kf_pro_sigma**2 / (2 * kf_theta), 1e-8)  # Ensure non-zero variance

    # ======== III. Perform Kalman Filtering ========
    n = len(series)
    filtered_states = np.zeros(n)
    variances = np.zeros(n)

    for t in range(n):
        observation = series.iloc[t]
        estimate_t, variance_t = make_prediction(observation, prior_estimate, prior_variance,
                                                 kf_mean, kf_theta, kf_obs_sigma, kf_pro_sigma)
        prior_estimate, prior_variance = estimate_t, variance_t

        filtered_states[t] = estimate_t
        variances[t] = variance_t

    # ======== IV. Convert to Series ========
    index = series.index
    filtered_states = pd.Series(filtered_states, index=index)
    variances = pd.Series(variances, index=index)
    
    return filtered_states, variances
