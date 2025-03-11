import sys
sys.path.append("../")
from Models import LinearRegression as reg
import auxiliary as aux

import numpy as np
import pandas as pd
import pywt

# ==================================================================================== #
# ======================= Unscaled Smoothed-like Series Features ===================== #
def average_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the different smoothed series =======
    rolling_average = price_series.rolling(window=window + 1).apply(lambda x: np.mean(x[:window]))

    # ======= II. Convert to pd.Series and Center =======
    rolling_average = (pd.Series(rolling_average, index=price_series.index) / (price_series + 1e-8)) - 1

    return rolling_average

# ____________________________________________________________________________________ #
def median_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the different smoothed series =======
    rolling_median = price_series.rolling(window=window + 1).apply(lambda x: np.median(x[:window]))

    # ======= II. Convert to pd.Series and Center =======
    rolling_median = (pd.Series(rolling_median, index=price_series.index) / (price_series + 1e-8)) - 1

    return rolling_median

# ____________________________________________________________________________________ #
def minimum_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the rolling maximum and minimum =======
    rolling_min = price_series.rolling(window=window + 1).apply(lambda x: np.min(x[:window]))

    # ======= II. Convert to pd.Series and Center =======
    rolling_min = (pd.Series(rolling_min, index=price_series.index) / (price_series + 1e-8)) - 1

    return rolling_min

# ____________________________________________________________________________________ #
def maximum_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the rolling maximum and minimum =======
    rolling_max = price_series.rolling(window=window + 1).apply(lambda x: np.max(x[:window]))

    # ======= II. Convert to pd.Series and Center =======
    rolling_max = (pd.Series(rolling_max, index=price_series.index) / (price_series + 1e-8)) - 1

    return rolling_max



# ==================================================================================== #
# ========================== Returns Distribution Features =========================== #
def volatility_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the rolling adjusted volatility =======
    returns_series = price_series.pct_change().dropna()
    rolling_vol = returns_series.rolling(window=window + 1).apply(lambda x: np.std(x[:window]))

    # ======= II. Convert to pd.Series and Center =======
    rolling_vol = pd.Series(rolling_vol, index=price_series.index)

    return rolling_vol

# ____________________________________________________________________________________ #
def skewness_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the rolling skewness =======
    returns_series = price_series.pct_change().dropna()
    rolling_skew = returns_series.rolling(window=window + 1).apply(lambda x: (x[:window]).skew())

    # ======= II. Convert to pd.Series and Center =======
    rolling_skew = pd.Series(rolling_skew, index=price_series.index)

    return rolling_skew

# ____________________________________________________________________________________ #
def kurtosis_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the rolling kurtosis =======
    returns_series = price_series.pct_change().dropna()
    rolling_kurt = returns_series.rolling(window=window + 1).apply(lambda x: (x[:window]).kurtosis())

    # ======= II. Convert to pd.Series and Center =======
    rolling_kurt = pd.Series(rolling_kurt, index=price_series.index)

    return rolling_kurt

# ____________________________________________________________________________________ #
def quantile_features(
    price_series: pd.Series,
    window: int,
    quantile: float,
):
    # ======= I. Compute the rolling quantile =======
    returns_series = price_series.pct_change().dropna()
    rolling_quantile = returns_series.rolling(window=window + 1).apply(lambda x: np.quantile(x[:window], quantile))

    # ======= II. Convert to pd.Series and Center =======
    rolling_quantile = pd.Series(rolling_quantile, index=price_series.index)

    return rolling_quantile



# ==================================================================================== #
# ============================= Series Trending Features ============================= #
def momentum_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the rolling momentum =======
    rolling_momentum = price_series.rolling(window=window + 1).apply(lambda x: aux.get_momentum(x[:window]))
    
    # ======= II. Convert to pd.Series and Center =======
    rolling_momentum = pd.Series(rolling_momentum, index=price_series.index)

    return rolling_momentum

# ____________________________________________________________________________________ #
def Z_momentum_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the rolling Z-momentum =======
    rolling_Z_momentum = price_series.rolling(window=window + 1).apply(lambda x: aux.get_Z_momentum(x[:window]))
    
    # ======= II. Convert to pd.Series and Center =======
    rolling_Z_momentum = pd.Series(rolling_Z_momentum, index=price_series.index)

    return rolling_Z_momentum

# ____________________________________________________________________________________ #
def linear_tempReg_features(
    price_series: pd.Series, 
    regression_window: int
):
    # ======= 0. Intermediate functions =======
    def compute_slope(series):
        _, coefficients, _ = aux.get_simple_TempReg(series)
        slope = coefficients[0]
        
        return slope

    def compute_T_stats(series):
        _, _, statistics = aux.get_simple_TempReg(series)
        T_stats = statistics['T_stats']
        
        return T_stats
    
    def compute_Pvalue(series):
        _, _, statistics = aux.get_simple_TempReg(series)
        P_value = statistics['P_values'][0]
        
        return P_value
    
    def compute_R_squared(series):
        _, _, statistics = aux.get_simple_TempReg(series)
        R_squared = statistics['R_squared']
        
        return R_squared

    # ======= I. Verify the price series is large enough =======
    if len(price_series) < regression_window:
        raise ValueError("Price series length must be greater than or equal to the regression window.")

    # ======= II. Compute the rolling regression statistics =======
    rolling_slope = price_series.rolling(window=regression_window + 1).apply(compute_slope, raw=False)
    rolling_tstat = price_series.rolling(window=regression_window + 1).apply(compute_T_stats, raw=False)
    rolling_pvalue = price_series.rolling(window=regression_window + 1).apply(compute_Pvalue, raw=False)
    rolling_r_squared = price_series.rolling(window=regression_window + 1).apply(compute_R_squared, raw=False)

    # ======= III. Convert to pd.Series and Unscale =======
    rolling_slope = pd.Series(rolling_slope, index=price_series.index) / (price_series + 1e-8)
    rolling_tstat = pd.Series(rolling_tstat, index=price_series.index)
    rolling_pvalue = pd.Series(rolling_pvalue, index=price_series.index)
    rolling_r_squared = pd.Series(rolling_r_squared, index=price_series.index)

    return rolling_slope, rolling_tstat, rolling_pvalue, rolling_r_squared

# ____________________________________________________________________________________ #
def nonlinear_tempReg_features(
    price_series: pd.Series,
    regression_window: int,
):
    # ======= 0. Intermediate functions =======
    def compute_slope(series):
        _, coefficients, _ = aux.get_quad_TempReg(series)
        slope = coefficients[0]
        
        return slope
    
    def compute_acceleration(series):
        _, coefficients, _ = aux.get_quad_TempReg(series)
        acceleration = coefficients[1]
        
        return acceleration

    def compute_T_stats(series):
        _, _, statistics = aux.get_quad_TempReg(series)
        T_stats = statistics['T_stats'][0]
        
        return T_stats
    
    def compute_Pvalue(series):
        _, _, statistics = aux.get_quad_TempReg(series)
        P_value = statistics['P_values'][0]
        
        return P_value
    
    def compute_R_squared(series):
        _, _, statistics = aux.get_quad_TempReg(series)
        R_squared = statistics['R_squared']
        
        return R_squared

    # ======= I. Verify the price series is large enough =======
    if len(price_series) < regression_window:
        raise ValueError("Price series length must be greater than or equal to the regression window.")

    # ======= II. Compute the rolling regression statistics =======
    rolling_slope = price_series.rolling(window=regression_window + 1).apply(compute_slope, raw=False)
    rolling_acceleration = price_series.rolling(window=regression_window + 1).apply(compute_acceleration, raw=False)
    rolling_tstat = price_series.rolling(window=regression_window + 1).apply(compute_T_stats, raw=False)
    rolling_pvalue = price_series.rolling(window=regression_window + 1).apply(compute_Pvalue, raw=False)
    rolling_r_squared = price_series.rolling(window=regression_window + 1).apply(compute_R_squared, raw=False)

    # ======= III. Convert to pd.Series and Unscale =======
    rolling_slope = pd.Series(rolling_slope, index=price_series.index) / (price_series + 1e-8)
    rolling_acceleration = pd.Series(rolling_acceleration, index=price_series.index) / (price_series + 1e-8)
    rolling_tstat = pd.Series(rolling_tstat, index=price_series.index)
    rolling_pvalue = pd.Series(rolling_pvalue, index=price_series.index)
    rolling_r_squared = pd.Series(rolling_r_squared, index=price_series.index)

    return rolling_slope, rolling_acceleration, rolling_tstat, rolling_pvalue, rolling_r_squared

# ____________________________________________________________________________________ #
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

        model = reg.MSERegression()
        model.fit(X, Y)
        
        hurst = model.coefficients[0]
        statistics = model.get_statistics()
        tstat = statistics['T_stats'][0]
        pvalue = statistics['P_values'][0]
        
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



# ==================================================================================== #
# ============================= Signal Processing Features ============================= #
def entropy_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Extract the signs series =======
    signs_series = aux.movements_signs(series=price_series)

    # ======= II. Compute the rolling entropy features =======
    rolling_shannon = signs_series.rolling(window=window + 1).apply(aux.get_shannon_entropy, raw=False)
    rolling_plugin = signs_series.rolling(window=window + 1).apply(aux.get_plugin_entropy, raw=False)
    rolling_lempel_ziv = signs_series.rolling(window=window + 1).apply(aux.get_lempel_ziv_entropy, raw=False)
    rolling_kontoyiannis = signs_series.rolling(window=window + 1).apply(aux.get_kontoyiannis_entropy, raw=False)

    # ======= III. Convert to pd.Series and Center =======
    rolling_shannon = pd.Series(rolling_shannon, index=price_series.index)
    rolling_plugin = pd.Series(rolling_plugin, index=price_series.index)
    rolling_lempel_ziv = pd.Series(rolling_lempel_ziv, index=price_series.index)
    rolling_kontoyiannis = pd.Series(rolling_kontoyiannis, index=price_series.index)

    return rolling_shannon, rolling_plugin, rolling_lempel_ziv, rolling_kontoyiannis

# ____________________________________________________________________________________ #
def wavelets_features(
    price_series: pd.Series,
    wavelet_window: int,
    wav_family: list = [],
    decomposition_level: int = 2,
):
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

