from ..measures import trending_measures as trend

import numpy as np
import pandas as pd



#! ==================================================================================== #
#! =========================== Series Smoothing Functions ============================= #
def average_smoothing(
    price_series: pd.Series,
    window: int,
) -> pd.Series:
    """
    Applies a simple moving average smoothing to the input series.
    
    Parameters:
        - price_series (pd.Series): Input series to be smoothed.
        - window (int): Window size for the moving average.
    
    Returns:
        - moving_average (pd.Series): Smoothed series.
    """
    # ======= I. Compute the moving average =======
    moving_average = price_series.rolling(window=window).mean()

    # ======= II. Convert to pd.Series and Normalize =======
    moving_average = pd.Series(moving_average, index=price_series.index)
    
    # ======= III. Change Name =======
    moving_average.name = f"MA_{window}"

    return moving_average

#*____________________________________________________________________________________ #
def ewma_smoothing(
    price_series: pd.Series, 
    window: int, 
    ind_lambda: float
) -> pd.Series:
    """
    Applies an Exponentially Weighted Moving Average (EWMA) smoothing to the input series.
    
    Parameters:
        - price_series (pd.Series): Input series to be smoothed.
        - window (int): Window size for the EWMA.
        - ind_lambda (float): Smoothing parameter (0 < ind_lambda < 1).
    
    Returns:
        - ewma (pd.Series): Smoothed series.
    """
    # ======= 0. Define the weighted moving average function =======
    def get_weightedMA(
        series: pd.Series, 
        weight_range: np.array
    ) -> np.array:
        """
        Computes the weighted moving average of a series using a specified weight range.
        
        Parameters:
            - series (pd.Series): Input series to be smoothed.
            - weight_range (np.array): Array of weights for the moving average.
        
        Returns:
            - wma (np.array): Weighted moving average of the series.
        """
        # ======= I. Check if the weights are valid =======
        values = np.array(series)
        values = values.astype("float64")
        wma = values.copy()

        if isinstance(weight_range, int):
            weights = np.array(range(1, weight_range + 1))
            rolling_window = weight_range
        else:
            weights = weight_range
            rolling_window = len(weight_range)

        # ======= II. Calculate the weighted moving average over a rolling window =======
        for i in range(0, len(values)):
            try:
                wma[i] = values[i - rolling_window + 1 : i + 1].dot(weights) / np.sum(weights)
            except:
                wma[i] = np.nan

        return wma
    
    # ======= I. Create the weights using a truncated exponential function =======
    exponential_weight_range = [(1 - ind_lambda) ** (i - 1) for i in range(1, window + 1)]
    exponential_weight_range.reverse()
    exponential_weight_range = np.array(exponential_weight_range)

    # ======= II. Perform the weighted moving average =======
    series = np.array(price_series)
    ewma = get_weightedMA(series=series, weight_range=exponential_weight_range)

    # ======= III. Convert to pd.Series =======
    ewma = pd.Series(ewma, index=price_series.index)
    
    # ======= IV. Change Name =======
    ewma.name = f"EWMA_{window}_{ind_lambda}"

    return ewma

#*____________________________________________________________________________________ #
def kalmanOU_smoothing(
    series: pd.Series, 
    smooth_coefficient: float
) -> tuple:
    """
    Applies a Kalman Filter to smooth the input series using an Ornstein-Uhlenbeck process.
    
    Parameters:
        - series (pd.Series): Input series to be smoothed.
        - smooth_coefficient (float): Coefficient for the Kalman Filter smoothing.
    
    Returns:
        - filtered_states (pd.Series): Smoothed series.
        - variances (pd.Series): Variances of the smoothed series.
    """
    # ======== 0. Define Kalman Filter Prediction Step ========
    def make_prediction(
        observation: float, 
        prior_estimate: float, 
        prior_variance: float,
        mean: float, 
        theta: float, 
        obs_sigma: float, 
        pro_sigma: float
    ) -> tuple:
        """
        Performs a Kalman Filter update step for the Ornstein-Uhlenbeck process.
        
        Parameters:
            - observation (float): Current observation.
            - prior_estimate (float): Prior estimate of the state.
            - prior_variance (float): Prior variance of the state.
            - mean (float): Mean of the process.
            - theta (float): Speed of mean reversion.
            - obs_sigma (float): Observation noise standard deviation.
            - pro_sigma (float): Process noise standard deviation.
        
        Returns:
            - estimate_t (float): Updated estimate of the state.
            - variance_t (float): Updated variance of the state.
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
    mu, theta, sigma, _ = trend.get_OU_estimation(series)
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

