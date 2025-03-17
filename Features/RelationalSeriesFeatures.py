import sys
sys.path.append("../")
from Features import auxiliary as aux

import pandas as pd
import numpy as np

# ==================================================================================== #
# =========================== Relationship Measures Features ========================= #
def cointegration_features(series_1: pd.Series, series_2: pd.Series, window: int):
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
        beta, intercept, adf_results, kpss_results, residuals = aux.cointegration_test(series_1=series1_window, series_2=series2_window)
        
        # II.3 Store Results
        beta_values[i] = beta
        intercept_values[i] = intercept
        adf_p_values[i] = adf_results[1]  # Extract p-value
        kpss_p_values[i] = kpss_results[1]  # Extract p-value
        residuals_values[i] = residuals[-1] # Store last residual
    
    # ======== III. Convert to Series ========
    index = series_1.index[window:]
    
    beta_series = pd.Series(beta_values, index=index)
    intercept_series = pd.Series(intercept_values, index=index)
    adf_p_values_series = pd.Series(adf_p_values, index=index)
    kpss_p_values_series = pd.Series(kpss_p_values, index=index)
    residuals_series = pd.Series(residuals_values, index=index)
    
    return beta_series, intercept_series, adf_p_values_series, kpss_p_values_series, residuals_series


# ==================================================================================== #
# ============================== Spread Series Features ============================== #
def ornstein_uhlenbeck_features(series_1: pd.Series, series_2: pd.Series, window: int, residuals_weights: np.array = None):
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
            _, _, _, _, residuals = aux.cointegration_test(series_1=series1_window, series_2=series2_window)
        else: 
            residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
        
        # II.3 Perform Ornstein-Uhlenbeck Estimation
        mu, theta, sigma, half_life = aux.ornstein_uhlenbeck_estimation(series=residuals)
        
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

# ____________________________________________________________________________________ #
def kalmanOU_features(series_1: pd.Series, series_2: pd.Series, window: int,  smooth_coefficient: float, residuals_weights: np.array = None):
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
            _, _, _, _, residuals = aux.cointegration_test(series_1=series1_window, series_2=series2_window)
        else: 
            residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
        
        # II.3 Perform Ornstein-Uhlenbeck Estimation
        filtered_states, variances = aux.kalmanOU_estimation(series=residuals, smooth_coefficient=smooth_coefficient)
        
        # II.4 Store Results
        state_values[i] = filtered_states[-1]
        variance_values[i] = variances[-1]
    
    # ======== III. Convert to Series ========
    index = series_1.index[window:]
    
    state_series = pd.Series(state_values, index=index)
    variance_series = pd.Series(variance_values, index=index)
    
    return state_series, variance_series