import os

os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.path.append("../")
from Models import LinearRegression as reg

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import warnings

# ==================================================================================== #
# ======================= Series Tendency Statistics Functions ======================= #
def get_momentum(series: pd.Series):
    first_value = series.iloc[0]
    last_value = series.iloc[-1]
    
    momentum = (last_value - first_value) / first_value
    
    return momentum

# ____________________________________________________________________________________ #
def get_Z_momentum(series: pd.Series):
    # ======= I. Compute Momentum =======
    momentum = get_momentum(series)
    
    # ======= II. Compute Standard Deviation of Returns =======
    returns_series = series.pct_change().dropna()
    returns_standard_deviation = np.std(returns_series)
    
    # ======= III. Compute Z-Momentum =======
    Z_momentum = momentum / returns_standard_deviation
    
    return Z_momentum

# ____________________________________________________________________________________ #
def get_simple_TempReg(series: pd.Series):
    # ======= I. Fit the temporal regression =======
    X = np.arange(len(series))
    model = reg.MSERegression()
    model.fit(X, series)
    
    # ======= II. Extract the coefficients and statistics =======
    coefficients = model.coefficients
    intercept = model.intercept
    
    statistics, residuals = model.get_statistics()

    return intercept, coefficients, statistics,residuals

# ____________________________________________________________________________________ #
def get_quad_TempReg(series: pd.Series):
    # ======= 1. Fit the temporal regression =======
    X = np.arange(len(series))
    X = np.column_stack((X, X**2))
    model = reg.MSERegression()
    model.fit(X, series)
    
    # ======= 2. Extract the coefficients and statistics =======
    coefficients = model.coefficients
    intercept = model.intercept
    
    statistics, residuals = model.get_statistics()

    return intercept, coefficients, statistics, residuals

# ____________________________________________________________________________________ #
def get_weightedMA(series: pd.Series, weight_range: np.array):
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
    
    
    
# ==================================================================================== #
# =============================== Series Entropy Functions =========================== #
def movements_signs(series: pd.Series):
    # ======= I. Compute the sign of the returns =======
    diff_series = series.diff()
    signs_series = np.sign(diff_series)

    # ======= II. Adjust for incorrect value =======
    signs_series.iloc[0] = 1
    signs_series[signs_series == 0] = np.nan
    signs_series = signs_series.ffill()

    return signs_series

# ____________________________________________________________________________________ #
def get_shannon_entropy(signs_series: pd.Series):
    # ======= I. Count the frequency of each symbol in the data =======
    _, counts = np.unique(signs_series, return_counts=True)

    # ======= II. Compute frequentist probabilities =======
    probabilities = counts / len(signs_series)

    # ======= III. Compute Shannon entropy =======
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

# ____________________________________________________________________________________ #
def get_plugin_entropy(signs_series: pd.Series, word_length: int = 1):
    # ======= 0. Auxiliary Function =======
    def compute_pmf(message: str, word_length: int):
        """
        This function computes the probability mass function for a one-dimensional discrete random variable.

        Args:
            message (str or array): Encoded message.
            word_length (int): Approximate word length.

        Returns:
            dict: Dictionary of the probability mass function for each word from the message.
        """
        # ======= I. Initialize the dictionary that will store the different words and their indexes of appearance =======
        unique_words_indexes = {}

        # ======= II. Iterate through the message to store the indexes of appareance of each different word =======
        for i in range(word_length, len(message)):
            word = message[i - word_length : i]
            word_index = i - word_length

            if word not in unique_words_indexes:
                # If the word is not in the dictionary, add it and store the index of appearance
                unique_words_indexes[word] = [word_index]

            else:
                # If the word is already in the dictionary, append the index of appearance
                unique_words_indexes[word] = unique_words_indexes[word] + [word_index]

        # ======= III. Compute the probability mass function =======
        total_count = float(len(message) - word_length)
        pmf = {word: len(unique_words_indexes[word]) / total_count for word in unique_words_indexes}

        return pmf

    # ======= I. Convert the signs series to a string =======
    message = signs_series.copy()
    message -= message.min()
    message = [int(x) for x in message]

    message = "".join(map(str, message))

    # ======= II. Compute the probability mass function =======
    pmf = compute_pmf(message, word_length)

    # ======= III. Compute the plug-in entropy =======
    entropy = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / word_length

    return entropy

#____________________________________________________________________________________ #
def get_lempel_ziv_entropy(signs_series: pd.Series):
    # ======= 0. Convert the signs series to a list =======
    signs_series_list = list(signs_series)
    series_size = len(signs_series_list)

    # ======= I. Initialize the variables =======
    complexity = 0
    patterns = []

    # ======= II. Iterate through the list to extract the different patterns =======
    index = 0
    index_extension = 1
    while index + index_extension <= series_size:
        if signs_series_list[index : index + index_extension] not in patterns:
            # II.1 Found new pattern : increment the complexity and add the pattern to the list
            patterns.append(signs_series_list[index : index + index_extension])
            complexity += 1

            # updates the index and the extension
            index += index_extension
            index_extension = 1
        else:
            # II.2 The current pattern was already seen : update the extension
            index_extension += 1

    # ======= III. Compute the Lempel-Ziv entropy =======
    entropy = complexity / series_size

    return entropy

# ____________________________________________________________________________________ #
def get_kontoyiannis_entropy(signs_series: pd.Series, window=None):
    # ======= 0. Auxiliary Functions =======
    def matchLength(message: str, starting_index: int, maximum_length: int):
        """
        This function computes the length of the longest substring that appears at least twice in the message.

        Args:
            message (str): The message to analyze.
            starting_index (int): The starting index of the substring.
            maximum_length (int): The maximum length of the substring.

        Returns:
            int: The length of the longest substring.
            str: The longest substring.
        """
        # ======= I. Initialize the longest sub string =======
        longest_substring = ""

        # ======= II. Iterate through the message to find the longest substring =======
        for possible_length in range(maximum_length):
            # II.1. Extract the maximum substring
            maximum_substring = message[starting_index : starting_index + possible_length + 1]

            # II.2. Iterate through the message to find the longest substring
            for index in range(starting_index - maximum_length, starting_index):
                # II.2.1. Extract the substring to compare
                substring = message[index : index + possible_length + 1]

                # II.2.2. Check if the substring is the longest
                if maximum_substring == substring:
                    longest_substring = maximum_substring
                    break

        # ======= III. Compute the length of the longest substring =======
        longest_substring_length = len(longest_substring) + 1

        return longest_substring_length, longest_substring

    # ======= I. Convert the signs series to a string =======
    out = {"nb_patterns": 0, "sum": 0, "patterns": []}
    message = signs_series.copy()
    message -= message.min()
    message = [int(x) for x in message]

    message = "".join(map(str, message))

    # ======= II. Extract the starting indexes (no need to iterate after half the series as a pattern would be found before if existing) =======
    if window is None:
        starting_indexes = range(1, len(message) // 2 + 1)
    else:
        window = min(window, len(message) // 2)
        starting_indexes = range(window, len(message) - window + 1)

    # ======= III. Compute the Kontoyiannis entropy =======
    for index in starting_indexes:
        # III.1. Compute the longest substring and its length
        if window is None:
            longest_pattern_length, longest_pattern = matchLength(message=message, starting_index=index, maximum_length=index)
            out["sum"] += np.log2(index + 1) / longest_pattern_length
        else:
            longest_pattern_length, longest_pattern = matchLength(message=message, starting_index=index, maximum_length=window)
            out["sum"] += np.log2(window + 1) / longest_pattern_length

        # III.2. Update the number of patterns and the list of patterns
        out["patterns"].append(longest_pattern)
        out["nb_patterns"] += 1

    # ======= IV. Compute the entropy and redundancy =======
    out["entropy"] = out["sum"] / out["nb_patterns"]
    out["redundancy"] = 1 - out["entropy"] / np.log2(len(message))

    entropy = out["entropy"] if out["entropy"] < 1 else 1

    return entropy



# ==================================================================================== #
# ============================ Series Relationship Functions ========================= #
def cointegration_test(series_1: pd.Series, series_2: pd.Series):
    # ======== I. Perform a Linear Regression ========
    model = reg.MSERegression()
    model.fit(series_2, series_1)

    # ======== II. Extract Regression Coefficients ========
    beta = model.coefficients[0]
    intercept = model.intercept

    # ======== III. Compute Residuals ========
    residuals = series_1 - (beta * series_2 + intercept)

    # ======== IV. Perform ADF & KPSS Tests ========
    adf_results = adfuller(residuals)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_results = kpss(residuals, regression="c", nlags="auto")

    return beta, intercept, adf_results, kpss_results, residuals

# ____________________________________________________________________________________ #
def ornstein_uhlenbeck_estimation(series: pd.Series):
    # ======== I. Initialize series ========
    series_array = np.array(series)
    differentiated_series = np.diff(series_array)
    mu = np.mean(series)
    
    X = series_array[:-1] - mu  # X_t - mu
    Y = differentiated_series  # X_{t+1} - X_t

    # ======== II. Perform OLS regression ========
    model = reg.MSERegression()
    model.fit(Y, X)
    
    # ======== III. Extract Parameters ========
    theta = -model.coefficients[0]
    if theta > 0:
        _, residuals = model.get_statistics()
        sigma = np.sqrt(np.var(residuals) * 2 * theta)
        half_life = np.log(2) / theta
    else:
        theta = 0
        sigma = 0
        half_life = 0

    return mu, theta, sigma, half_life

# ____________________________________________________________________________________ #
def kalmanOU_estimation(series: pd.Series, smooth_coefficient: float):
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