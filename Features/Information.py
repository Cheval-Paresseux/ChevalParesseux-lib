"""
# Description: This file contains the functions used to validate the cointegration between a pair of assets and to compute the useful metrics for the strategy.
_____
generate_combinations: generates unique combinations of assets from the dataframe, limiting overlap between combinations.
                        -> The limitation of overlap is controlled by the max_shared_assets parameter, it could lead to an unstability of the combinations computed if the parameter is too low.
_____
cointegration_test: performs a linear regression on the DataFrame's columns and ADF/KPSS test on the residuals to check for stationarity.
estimate_ou_parameters: estimates the parameters of an Ornstein-Uhlenbeck process given a time series (we expect the time series to be stationary).
_____
ou_process_probability: computes the probability of the Ornstein-Uhlenbeck process to cross a certain threshold.
get_expected_return: computes the expected return of a pairs inside the framework of an Ornstein-Uhlenbeck process and Long/Short position.
_____
OU_KalmanFilter: class to perform the Kalman filter on the residuals of the spread, it gives an estimation of the actual spread.
_____
POTENTIAL IMPROVEMENTS:
    - Add more/Improve tests to validate the cointegration between a pair of assets.
    - Add more/Improve metrics to compute the expected return of a pair.
    - Add more functionalities to the Kalman filter as generalized ARMA models.
    - Apply Machine Learning models to predict the spread.
"""

import os

os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
from scipy.stats import norm

import warnings

warnings.filterwarnings("ignore")


# ========================================================================================================== #
def generate_combinations(
    df: pd.DataFrame,
    num_assets_per_comb: int,
    max_shared_assets: int,
):
    """
    Generate unique combinations of assets from the dataframe, limiting overlap between combinations.
    Allows certain assets to appear in up to a specified number of combinations.

    Args:
        df (pd.DataFrame): DataFrame of asset price histories.
        num_assets_per_comb (int): Number of assets per combination.
        max_shared_assets (int): Maximum times an asset can be shared across different combinations.

    Returns:
        list_of_dfs (list[pd.DataFrame]): List of DataFrames, each containing one valid combination of assets.
    """
    # ======== I. Compute all possible combinations of assets ========
    assets = df.columns.tolist()
    all_combinations = list(itertools.combinations(assets, num_assets_per_comb))

    # ======== II. Track how many times each asset appears in selected combinations ========
    asset_count = {asset: 0 for asset in assets}
    selected_combinations = []

    for combination in all_combinations:
        # -------- Check overlap with already selected combinations
        is_valid_combination = True
        for asset in combination:
            if asset_count[asset] >= max_shared_assets:
                is_valid_combination = False
                break

        # --------- If valid, add the combination and update asset counts
        if is_valid_combination:
            selected_combinations.append(combination)
            for asset in combination:
                asset_count[asset] += 1

    # ======== III. Storing each combination in a list of DataFrames ========
    list_of_dfs = [df[list(combination)] for combination in selected_combinations]

    return list_of_dfs


# ========================================================================================================== #
def cointegration_test(df: pd.DataFrame):
    """
    Perform a linear regression on the DataFrame's columns.
    If the DataFrame has n columns, the first column is the dependant variable and the rest are independant variables.
    Perform as well an ADF/KPSS test on the residuals to check for stationarity.

    Args:
        df (pd.DataFrame): DataFrame of asset price histories

    Returns:
        tuple: Tuple containing the coefficients of the linear regression, the ADF/KPSS test results, and the residuals.
    """
    # ======== I. Performs the linear regression ========
    log_df = np.log(
        df
    )  # Apply log transformation to the prices as we model the spread as a linear combination of the assets' log prices

    X = log_df.iloc[:, 1:]  # Independant variable(s)
    Y = log_df.iloc[:, 0]  # Dependant variable
    model = LinearRegression()
    model.fit(X, Y)

    # ======== II. Store the results ========
    coefficients = (
        -model.coef_
    )  # X - bY - intercept = epsilon -> that's why apply a minus sign
    intercept = -model.intercept_

    coefficients_total = np.append(
        [1], coefficients
    )  # Add a 1 in front of the coefficients for the target variable

    coefficients_with_intercept = np.append([intercept], coefficients_total)

    # ---------
    residuals = log_df.dot(coefficients_total) + intercept

    # ---------
    adf_results = adfuller(residuals)
    kpss_results = kpss(residuals, regression="c")

    return coefficients_with_intercept, adf_results, kpss_results, residuals


# ---------------------------------------------------------------------------------------------------------- #
def estimate_ou_parameters(data: np.array):
    """
    Estimate the parameters of an Ornstein-Uhlenbeck process given a time series.

    Parameters:
        data (array-like): A time series of observations (daily observations).

    Returns:
        dict: Estimated parameters of the OU process (mu, theta, sigma, half_life).
    """
    # ======== I. Perform OLS on AR(1) ========
    mu = np.mean(data)
    delta_data = np.diff(data)

    # ---------
    X = data[:-1] - mu  # X_t - mu
    Y = delta_data  # X_{t+1} - X_t

    reg = sm.OLS(Y, X).fit()

    # ======== II. Estimate parameters ========
    theta = -reg.params[0]
    if theta > 0:
        residuals = reg.resid
        sigma = np.sqrt(np.std(residuals) * 2 * theta)
        half_life = np.log(2) / theta
    else:
        theta = 0
        sigma = 0
        half_life = 0

    return mu, theta, sigma, half_life


# ========================================================================================================== #
def ou_process_probability(
    last_value: float,
    up_threshold: float,
    half_life: float,
    mu: float,
    theta: float,
    sigma: float,
):
    """
    Compute the probability of the Ornstein-Uhlenbeck process to cross a certain threshold.

    Args:
        last_value (float): Last value of the process.
        up_threshold (float): Threshold to cross.
        half_life (float): Half-life of the process.
        mu (float): Mean of the process.
        theta (float): Speed of mean reversion.
        sigma (float): Standard deviation of the process.

    Returns:
        float: Probability of the process to cross the threshold.
    """
    # Mean of X_{t+z}
    mean_xtz = mu + (last_value - mu) * np.exp(-theta * half_life)

    # Variance of X_{t+z}
    var_xtz = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * half_life))

    # Standardized Z
    z_score = (up_threshold - mean_xtz) / np.sqrt(var_xtz)

    # Probability P(X_{t+z} > y)
    prob = 1 - norm.cdf(z_score)

    return prob


# ---------------------------------------------------------------------------------------------------------- #
def get_expected_return(
    normalized_weights: list,
    z_score: float,
    half_life: float,
    theta: float,
    sigma: float,
    mu: float,
    gap: float,
    Zup_exit_threshold: float,
    leverage: bool,
    risk_free_rate: float,
    cash_margin: float,
    collateralization_level,
    haircut: float,
):
    """
    Compute the expected return of a combination inside the framework of a Pairs Tarding Like Strategy.

    Args:
        normalized_weights (list): Normalized weights of the pair.
        z_score (float): Z-score of the pair.
        half_life (float): Half-life of the pair.
        gap (float): Gap between the two assets.
        leverage (bool): Whether the strategy uses leverage.
        risk_free_rate (float): Risk-free rate.
        cash_margin (float): Cash margin.
        collateralization_level (float): Collateralization level.
        haircut (float): Haircut.

    Returns:
        float: Expected return of the pair.
    """
    # ========== 1. Compute the cash needed to enter the trade  ========== (This takes into account that we are long and short the assets so we need to collateralize the short position)
    cash_needed = 0
    if z_score < 0:
        # --------- We long the pair
        short_position = sum(
            abs(short_value) for short_value in normalized_weights if short_value < 0
        )
        long_position = sum(
            long_value for long_value in normalized_weights if long_value > 0
        )
        cash_for_deposit = max(
            short_position * collateralization_level - long_position * (1 - haircut),
            0,
        )
        cash_needed = long_position - short_position + cash_for_deposit

    elif z_score > 0:
        # --------- We short the pair
        short_position = sum(
            short_value for short_value in normalized_weights if short_value > 0
        )
        long_position = sum(
            abs(long_value) for long_value in normalized_weights if long_value < 0
        )
        cash_for_deposit = max(
            short_position * collateralization_level - long_position * (1 - haircut),
            0,
        )
        cash_needed = long_position - short_position + cash_for_deposit

    else:
        # Neutralizing the pairs with a z-score of 0 because it means the pair is not stationary (OU process)
        cash_needed = 1

    # ========== 2. Compute the expected return of the trade ==========
    if leverage is False:
        leverage_ratio = 1
    else:
        leverage_ratio = (1 - cash_margin) / abs(cash_needed)

    if half_life != 0:
        up_threshold = Zup_exit_threshold * sigma
        lose_gap = gap - up_threshold
        probability_cross_exit_threshold = ou_process_probability(
            last_value=gap,
            up_threshold=up_threshold,
            half_life=2 * half_life,
            mu=mu,
            theta=theta,
            sigma=sigma,
        )
        expectancy = lose_gap * probability_cross_exit_threshold + (
            gap - gap * np.exp(-theta * 2 * half_life)
        ) * (1 - probability_cross_exit_threshold)

        available_cash_expected_return = cash_margin * (
            (1 + risk_free_rate) ** (half_life) - 1
        )

        expected_return = expectancy + available_cash_expected_return
        expected_return = expected_return / (2 * half_life)
    else:
        expected_return = 0

    return expected_return, cash_needed, leverage_ratio


# ---------------------------------------------------------------------------------------------------------- #
def trading_score(nb_operations, min_nb_operations, target_nb_operations, maturity):
    """
    Compute the trading score of a strategy based on the number of operations made.

    The score evaluates the performance of a strategy by comparing the number of operations made
    over a given period (adjusted for maturity) to a predefined minimum and target range.
    It uses a piecewise function to reflect three performance tiers:
    insufficient activity (score = 0), growth phase (concave scoring), and sufficient activity
    (logarithmic adjustment).

    Args:
        nb_operations (int): Total number of operations made by the strategy.
        min_nb_operations (int): Minimum threshold of operations required to start scoring.
        target_nb_operations (int): Target number of operations for optimal scoring.
        maturity (float): Time period over which the operations are measured.

    Returns:
        trading_score (float): A value between 0 and 1 representing the trading score of the strategy.
    """
    # ======= I. Normalize operations over the time period =======
    nb_operations_time_adjusted = nb_operations / maturity

    # ======= II. Compute the trading score using piecewise logic =======
    if nb_operations_time_adjusted <= min_nb_operations:
        # Case 1: Insufficient activity
        trading_score = 0

    elif nb_operations_time_adjusted <= target_nb_operations:
        # Case 2: Growth phase
        # It starts at 0.4 (just above the minimum threshold) and grows to approximately 0.8.
        diff = nb_operations_time_adjusted - min_nb_operations
        interval = target_nb_operations - min_nb_operations
        trading_score = 0.4 + 0.5 * np.sqrt(diff / interval)  # Concave growth

    else:
        # Case 3: sufficient activity
        trading_score = (
            0.8 + np.log(3 + nb_operations_time_adjusted - target_nb_operations) / 10
        )

    # ======= III. Ensure non-negativity =======
    # The logic already ensures that the score is non-negative, but apply max(0, score) as a safeguard.
    trading_score = max(0, trading_score)

    return trading_score


# ========================================================================================================== #
class OU_KalmanFilter:
    def __init__(
        self, mean: float, theta: float, obs_sigma: float, smooth_coefficient: float
    ):
        """
        Initialize the Kalman filter for the Ornstein-Uhlenbeck process.

        Args:
            residuals (pd.Series): The residuals of the spread.
            mean (float): The mean of the spread.
            theta (float): The speed of mean reversion.
            obs_sigma (float): The standard deviation of the observation noise.
            smooth_coefficient (float): The smoothing coefficient.
        """
        self.mean = mean
        self.theta = theta
        self.obs_sigma = obs_sigma
        self.pro_sigma = obs_sigma * smooth_coefficient

        # Initialize Kalman filter states
        self.prior_estimate = self.mean
        self.prior_variance = self.pro_sigma**2 / (2 * self.theta)

    # ---------------------------------------------------------------------------------------------------------------#
    def make_prediction(self, observation: float):
        """
        Make a prediction using the Kalman filter.
        Args:
            observation (float): The observed spread at time t.

        Returns:
            estimate_t (float): The filtered state estimate at time t.
            variance_t (float): The filtered state variance at time t.
        """
        # ======= I. Initialize state and variance =======
        estimate_t = self.prior_estimate
        variance_t = self.prior_variance

        # ======= II. Observation update =======
        innovation_t = observation - estimate_t
        innovation_variance_t = variance_t + self.obs_sigma**2
        kalman_gain_t = variance_t / innovation_variance_t

        # ======= III. Update state and variance =======
        estimate_t = estimate_t + kalman_gain_t * innovation_t
        variance_t = (1 - kalman_gain_t) * variance_t

        # ======= IV. Prediction step (OU transition) =======
        estimate_t = self.mean + (1 - self.theta) * (estimate_t - self.mean)
        variance_t = (1 - self.theta) ** 2 * variance_t + self.pro_sigma**2

        # ======= V. Store results =======
        self.prior_estimate = estimate_t
        self.prior_variance = variance_t

        return estimate_t, variance_t

    # ---------------------------------------------------------------------------------------------------------------#
    def series_filter(self, residuals: pd.Series):
        """
        Perform the Kalman filter on a series of residuals.

        Args:
            residuals (pd.Series): The residuals of the spread.

        Returns:
            filtered_states (pd.Series): The filtered state estimates.
            variances (pd.Series): The filtered state variances.
        """
        # ======= I. Initialize the vectors =======
        n = len(residuals)
        filtered_states = np.zeros(n)
        variances = np.zeros(n)

        # ======= II. Run the Kalman filter =======
        for t in range(n):
            # Observation update
            observation = residuals.iloc[t]
            estimate_t, variance_t = self.make_prediction(observation)

            # Store results
            filtered_states[t] = estimate_t
            variances[t] = variance_t

        return pd.Series(filtered_states, index=residuals.index), pd.Series(
            variances, index=residuals.index
        )
