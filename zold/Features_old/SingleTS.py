import os
import sys

sys.path.append(os.path.abspath("../"))
import Smoothing.NoLook_Filters as filters

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as sps
import pywt


# ========================== Single TS Features ==========================
def minMax_features(
    price_series: pd.Series,
    window: int,
):
    """
    This function computes the rolling minimum and maximum of a price series and compares it to the price series.

    Args:
        price_series (pd.Series): The price series of the asset.
        window (int): The window size for the rolling minimum and maximum.

    Returns:
        pd.Series: The rolling minimum of the price series.
        pd.Series: The rolling maximum of the price series.
    """
    # ======= I. Compute the rolling maximum and minimum =======
    rolling_min = price_series.rolling(window=window + 1).apply(lambda x: np.min(x[:window]))
    rolling_max = price_series.rolling(window=window + 1).apply(lambda x: np.max(x[:window]))

    # ======= II. Convert to pd.Series and Center =======
    rolling_min = (pd.Series(rolling_min, index=price_series.index) / (price_series + 1e-8)) - 1
    rolling_max = (pd.Series(rolling_max, index=price_series.index) / (price_series + 1e-8)) - 1

    return rolling_min, rolling_max


# -----------------------------------------------------------------------------
def smoothing_features(
    price_series: pd.Series,
    window: int,
    ind_lambda: float
):
    """
    This function computes different smoothed series of a price series and compares it to the price series.

    Args:
        price_series (pd.Series): The price series of the asset.
        window (int): The window size for the rolling mean.

    Returns:
        pd.Series: The rolling mean of the price series.
    """
    # ======= I. Compute the different smoothed series =======
    rolling_average = price_series.rolling(window=window + 1).apply(lambda x: np.mean(x[:window]))
    rolling_ewma = filters.exponential_weighted_moving_average(price_series=price_series, window=window, ind_lambda=ind_lambda)

    # ======= II. Convert to pd.Series and Center =======
    rolling_average = (pd.Series(rolling_average, index=price_series.index) / (price_series + 1e-8)) - 1
    rolling_ewma = (pd.Series(rolling_ewma, index=price_series.index) / (price_series + 1e-8)) - 1

    return rolling_average, rolling_ewma


# -----------------------------------------------------------------------------
def volatility_features(
    price_series: pd.Series,
    window: int,
):
    """
    This function computes the rolling volatility of a price series and compares it to the price series.

    Args:
        price_series (pd.Series): The price series of the asset.
        window (int): The window size for the rolling volatility.

    Returns:
        pd.Series: The rolling volatility of the price series.
    """
    # ======= I. Compute the rolling adjusted volatility =======
    returns_series = price_series.pct_change().dropna()
    rolling_vol = returns_series.rolling(window=window + 1).apply(lambda x: np.std(x[:window]))

    # ======= II. Convert to pd.Series and Center =======
    rolling_vol = pd.Series(rolling_vol, index=price_series.index)
    rolling_vol_mean = rolling_vol.rolling(window=252).mean()

    rolling_vol = rolling_vol - rolling_vol_mean

    return rolling_vol


# -----------------------------------------------------------------------------
def momentum_features(
    price_series: pd.Series,
    window: int,
):
    """
    This function computes the rolling Z-score momentum of a price series and compares it to the price series.

    Args:
        price_series (pd.Series): The price series of the asset.
        window (int): The window size for the rolling momentum.

    Returns:
        pd.Series: The rolling momentum of the price series.
        pd.Series: The rolling Z-score momentum of the price series.
    """
    # ======= I. Compute the rolling momentum =======
    returns_series = price_series.pct_change().dropna()
    rolling_vol = returns_series.rolling(window=window + 1).apply(lambda x: np.std(x[:window]))

    rolling_momentum = price_series.rolling(window=window + 1).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])

    # ======= II. Convert to pd.Series and Center =======
    rolling_momentum = pd.Series(rolling_momentum, index=price_series.index)

    # ======= III. Compute the Z-score momentum =======
    rolling_Z_momentum = rolling_momentum / rolling_vol

    return rolling_momentum, rolling_Z_momentum


# -----------------------------------------------------------------------------
def linear_tempReg_features(
    price_series: pd.Series, 
    regression_window: int
):
    """
    Computes the rolling trend and t-statistic of a price series.

    Args:
        price_series (pd.Series): The price series of the asset.
        regression_window (int): The window size for the rolling regression.

    Returns:
        pd.Series: The rolling trend of the price series (normalized).
        pd.Series: The rolling t-statistic of the trend.
    """

    # ======= 0. Intermediate functions =======
    def get_trend_and_tstat(series):
        # ------- 1. Fit the OLS regression -------
        X = np.arange(len(series))
        X = sm.add_constant(X)  # Add intercept for OLS
        model = sm.OLS(series, X, missing="drop")
        results = model.fit()

        # ------- 2. Extract the trend coefficient and t-statistic -------
        trend_coefficient = results.params[1]
        t_statistic = results.tvalues[1]

        return trend_coefficient, t_statistic

    def compute_trend(series):
        trend, _ = get_trend_and_tstat(series)
        return trend

    def compute_tstat(series):
        _, t_stat = get_trend_and_tstat(series)
        return t_stat

    # ======= I. Initialize the trend, acceleration, and tstats =======
    if len(price_series) < regression_window:
        raise ValueError("Price series length must be greater than or equal to the regression window.")

    # ======= II. Compute the rolling trend, acceleration, and tstats =======
    rolling_trend = price_series.rolling(window=regression_window + 1).apply(compute_trend, raw=False)
    rolling_tstat = price_series.rolling(window=regression_window + 1).apply(compute_tstat, raw=False)

    # ======= III. Convert to pd.Series and Center =======
    rolling_trend = pd.Series(rolling_trend, index=price_series.index) / (price_series + 1e-8)
    rolling_tstat = pd.Series(rolling_tstat, index=price_series.index)

    rolling_tstat_mean = rolling_tstat.rolling(window=252).mean()
    rolling_tstat = rolling_tstat - rolling_tstat_mean

    return rolling_trend, rolling_tstat


# -----------------------------------------------------------------------------
def nonlinear_tempReg_features(
    price_series: pd.Series,
    regression_window: int,
):
    """
    This function computes the rolling trend, acceleration, and t-statistic of a price series and compares it to the price series.

    Args:
        price_series (pd.Series): The price series of the asset.
        regression_window (int): The window size for the rolling trend.

    Returns:
        pd.Series: The rolling trend of the price series.
        pd.Series: The rolling acceleration of the price series.
        pd.Series: The rolling t-statistic of the price series.
    """

    # ======= 0. Intermediate functions =======
    def get_trend_acceleration_and_tstat(series: pd.Series):
        # ------- 1. Fit the OLS regression -------
        X = np.arange(len(series))
        X_quad = np.column_stack((X, X**2))
        X_quad = sm.add_constant(X_quad)
        model = sm.OLS(series, X_quad, missing="drop")
        results = model.fit()

        # ------- 2. Extract the trend coefficient, acceleration coefficient, and t-statistic -------
        trend_coefficient = results.params[1]
        acceleration_coefficient = results.params[2]
        t_statistic = results.tvalues[2]

        return trend_coefficient, acceleration_coefficient, t_statistic

    def compute_trend(series):
        trend, _, _ = get_trend_acceleration_and_tstat(series)
        return trend

    def compute_acceleration(series):
        _, acceleration, _ = get_trend_acceleration_and_tstat(series)
        return acceleration

    def compute_tstat(series):
        _, _, tstat = get_trend_acceleration_and_tstat(series)
        return tstat

    # ======= I. Initialize the trend, acceleration, and tstats =======
    if len(price_series) < regression_window:
        raise ValueError("Price series length must be greater than or equal to the regression window.")

    # ======= II. Compute the rolling trend, acceleration, and tstats =======
    nonLin_rolling_trend = price_series.rolling(window=regression_window + 1).apply(compute_trend, raw=False)
    nonLin_rolling_acceleration = price_series.rolling(window=regression_window + 1).apply(compute_acceleration, raw=False)
    nonLin_rolling_tstat = price_series.rolling(window=regression_window + 1).apply(compute_tstat, raw=False)

    # ======= III. Convert to pd.Series and Center =======
    nonLin_rolling_trend = pd.Series(nonLin_rolling_trend, index=price_series.index) / (price_series + 1e-8)
    nonLin_rolling_acceleration = pd.Series(nonLin_rolling_acceleration, index=price_series.index) / (price_series + 1e-8)
    nonLin_rolling_tstat = pd.Series(nonLin_rolling_tstat, index=price_series.index)

    nonLin_rolling_tstat_mean = nonLin_rolling_tstat.rolling(window=252).mean()
    nonLin_rolling_tstat = nonLin_rolling_tstat - nonLin_rolling_tstat_mean

    return nonLin_rolling_trend, nonLin_rolling_acceleration, nonLin_rolling_tstat


# -----------------------------------------------------------------------------
def hurst_exponent_features(
    price_series: pd.Series, 
    power: int
):
    # ======= 0. Initialize the variables =======
    prices_array = np.array(price_series)
    returns_array = prices_array[1:] / prices_array[:-1] - 1

    n = 2**power

    hursts = np.array([])
    tstats = np.array([])
    pvalues = np.array([])

    # ======= 1. Compute the Hurst Exponent =======
    for t in np.arange(n, len(returns_array) + 1):
        data = returns_array[t - n : t]
        X = np.arange(2, power + 1)
        Y = np.array([])

        for p in X:
            m = 2**p
            s = 2 ** (power - p)
            rs_array = np.array([])

            for i in np.arange(0, s):
                subsample = data[i * m : (i + 1) * m]
                mean = np.average(subsample)
                deviate = np.cumsum(subsample - mean)
                difference = max(deviate) - min(deviate)
                stdev = np.std(subsample)
                rescaled_range = difference / stdev
                rs_array = np.append(rs_array, rescaled_range)

            Y = np.append(Y, np.log2(np.average(rs_array)))

        reg = sm.OLS(Y, sm.add_constant(X))
        res = reg.fit()
        hurst = res.params[1]
        tstat = (res.params[1] - 0.5) / res.bse[1]
        pvalue = 2 * (1 - sps.t.cdf(abs(tstat), res.df_resid))
        hursts = np.append(hursts, hurst)
        tstats = np.append(tstats, tstat)
        pvalues = np.append(pvalues, pvalue)

    # ======= 2. Convert to pd.Series and Center =======
    hursts = pd.Series([np.nan] * n + list(hursts), index=price_series.index) - 0.5
    tstats = pd.Series([np.nan] * n + list(tstats), index=price_series.index)
    pvalues = pd.Series([np.nan] * n + list(pvalues), index=price_series.index)

    tstats_mean = tstats.rolling(window=252).mean()
    tstats = tstats - tstats_mean

    pvalues_mean = pvalues.rolling(window=252).mean()
    pvalues = pvalues - pvalues_mean

    return hursts, tstats, pvalues


# -----------------------------------------------------------------------------
def entropy_features(
    price_series: pd.Series,
    window: int,
):
    # ======= 0. Auxiliary function =======
    def movements_signs(price_series: pd.Series):
        """
        This function computes the sign of the returns of a price series.

        Args:
            price_series (pd.Series): The price series of the asset.

        Returns:
            pd.Series: The sign of the returns of the price series.
        """
        # ======= I. Compute the sign of the returns =======
        prices = price_series.diff()
        prices = np.sign(prices)

        # ======= II. Adjust for incorrect value =======
        prices.iloc[0] = 1
        prices[prices == 0] = np.nan
        prices = prices.ffill()

        return prices

    # ======= I. Extract the signs series =======
    signs_series = movements_signs(price_series=price_series)

    # ======= II. Compute the rolling entropy features =======
    rolling_shannon = signs_series.rolling(window=window + 1).apply(get_shannon_entropy, raw=False)
    rolling_plugin = signs_series.rolling(window=window + 1).apply(get_plugin_entropy, raw=False)
    rolling_lempel_ziv = signs_series.rolling(window=window + 1).apply(get_lempel_ziv_entropy, raw=False)
    rolling_kontoyiannis = signs_series.rolling(window=window + 1).apply(get_kontoyiannis_entropy, raw=False)

    # ======= III. Convert to pd.Series and Center =======
    rolling_shannon = pd.Series(rolling_shannon, index=price_series.index)
    rolling_plugin = pd.Series(rolling_plugin, index=price_series.index)
    rolling_lempel_ziv = pd.Series(rolling_lempel_ziv, index=price_series.index)
    rolling_kontoyiannis = pd.Series(rolling_kontoyiannis, index=price_series.index)

    return rolling_shannon, rolling_plugin, rolling_lempel_ziv, rolling_kontoyiannis


# -----------------------------------------------------------------------------
def wavelets_features(
    price_series: pd.Series,
    wavelet_window: int,
    wav_family: list,
    decomposition_level: int,
):
    """
    This function computes the wavelet features of a price series.

    Args:
        price_series (pd.Series): The price series of the asset.
        wavelet_window (int): The window size for the wavelet decomposition.
        wav_family (list): The list of wavelet families to compute the features.
        decomposition_level (int): The decomposition level for the wavelet transform.

    Returns:
        tuple: The wavelet features of the price series.
    """
    # ======= 0. Initialize the input series as a dataframe to store the wavelets =======
    if len(wav_family) == 0:
        wav_family = ["haar", "db1", "db2", "db3", "db4", "sym2", "sym3", "sym4", "sym5", "coif1", "coif2", "coif3", "coif4", "bior1.1", "bior1.3", "bior1.5", "bior2.2", "rbio1.1", "rbio1.3", "rbio1.5"]

    if price_series.name is None:
        price_series.name = "close"
    price_df = price_series.to_frame().copy()
    price_df.rename(columns={price_series.name: "close"}, inplace=True)

    # ======= I. Compute the wavelets for each family =======
    for wavelet in wav_family:
        # I.1 Initialize the lists to store the wavelet features
        mean = [[None] * wavelet_window for _ in range(decomposition_level)]
        median = [[None] * wavelet_window for _ in range(decomposition_level)]
        std = [[None] * wavelet_window for _ in range(decomposition_level)]
        max = [[None] * wavelet_window for _ in range(decomposition_level)]
        min = [[None] * wavelet_window for _ in range(decomposition_level)]
        # => Each inside list corresponds to a decomposition level, and each element of those inside lists corresponds to a window

        # I.2 Compute the wavelet features
        for index in range(wavelet_window, price_df.shape[0]):
            # I.2.i Extract the rolling window of the price series and compute the wavelet coefficients
            price_window = price_df.iloc[index - wavelet_window : index, 0].copy()
            coeffs = pywt.wavedec(
                price_window,
                wavelet,
                level=decomposition_level,
            )

            # I.2.ii Compute the wavelet features for each decomposition level
            for level in range(decomposition_level):
                # We start at level 1 because the first element of the coeffs list is the approximation coefficients
                mean[level].append(np.mean(coeffs[level + 1]))
                median[level].append(np.median(coeffs[level + 1]))
                std[level].append(np.std(coeffs[level + 1]))
                max[level].append(np.max(coeffs[level + 1]))
                min[level].append(np.min(coeffs[level + 1]))

        # I.3 Store the wavelet features in the dataframe
        for level in range(decomposition_level):
            price_df.loc[:, f"{wavelet}_{level + 1}_mean"] = mean[level]
            price_df.loc[:, f"{wavelet}_{level + 1}_median"] = median[level]
            price_df.loc[:, f"{wavelet}_{level + 1}_std"] = std[level]
            price_df.loc[:, f"{wavelet}_{level + 1}_max"] = max[level]
            price_df.loc[:, f"{wavelet}_{level + 1}_min"] = min[level]

    #  ======= II. Convert to Series and Center =======
    price_df.drop(labels="close", axis=1, inplace=True)
    features_columns = price_df.columns

    for feature in features_columns:
        price_df[feature] = price_df[feature] - price_df[feature].rolling(window=252).mean()

    features_tuple = tuple([price_df[feature] for feature in features_columns])

    return features_tuple


# =========================================================================
# ========================== Auxiliary functions ==========================
def get_shannon_entropy(signs_series: pd.Series):
    """
    This function computes the Shannon entropy of a series of signs.

    Args:
        signs_series (pd.Series): The series of signs (must be a series of 1s and -1s).

    Returns:
        float: The Shannon entropy of the series of signs.
    """
    # ======= I. Count the frequency of each symbol in the data =======
    _, counts = np.unique(signs_series, return_counts=True)

    # ======= II. Compute frequentist probabilities =======
    probabilities = counts / len(signs_series)

    # ======= III. Compute Shannon entropy =======
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


# -----------------------------------------------------------------------------
def get_plugin_entropy(signs_series: pd.Series, word_length: int = 1):
    """
    This function computes the plug-in entropy estimator.

    Args:
        signs_series (pd.Series): The series of signs (must be a series of 1s and -1s).
        word_length (int): The approximate word length.

    Returns:
        float: The plug-in entropy.
    """

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


# -----------------------------------------------------------------------------
def get_lempel_ziv_entropy(signs_series: pd.Series):
    """
    This function computes the Lempel-Ziv entropy of a series of signs.

    Args:
        signs_series (pd.Series): The series of signs (must be a series of 1s and -1s).

    Returns:
        float: The Lempel-Ziv entropy of the series of signs.
    """

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


# -----------------------------------------------------------------------------
def get_kontoyiannis_entropy(signs_series: pd.Series, window=None):
    """
    This function computes the Kontoyiannis entropy of a series of signs.

    Args:
        signs_series (pd.Series): The series of signs (must be a series of 1s and -1s).
        window (int): The window size for the rolling entropy.

    Returns:
        float: The Kontoyiannis entropy of the series of signs
    """

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