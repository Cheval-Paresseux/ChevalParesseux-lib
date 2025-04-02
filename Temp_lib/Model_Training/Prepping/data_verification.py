from ..Prepping import common as com

import os

os.environ["MKL_NUM_THREADS"] = "1"
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
import pandas as pd
import numpy as np


#! ==================================================================================== #
#! ================================ Features Check ==================================== #
def check_for_error_values(
    feature_series: pd.Series
) -> tuple:
    """
    This function checks for error values in a given feature series. 
    It identifies NaN values, infinite values, and calculates the proportion of error values. 
    It also creates a clean series by replacing infinite values with NaN and filling NaN values using forward fill method.
    Parameters:
        - feature_series (pd.Series): The feature series to be checked for error values.
    
    Returns:
        - clean_series (pd.Series): The clean series with infinite values replaced by NaN and NaN values filled.
        - error_proportion (float): The proportion of error values in the series.
        - beginning_nans (list): List of indexes where NaN values occur before the first valid index.
        - middle_nans (list): List of indexes where NaN values occur after the first valid index.
        - infinite_indexes (list): List of indexes where infinite values occur.
    """
    # ======= I. Duplicate the series =======
    auxiliary_series = feature_series.copy()
    
    # ======= II. Identify NaN values =======
    nan_indexes = auxiliary_series[auxiliary_series.isna()].index
    
    # ======= III. Identify the first valid index =======
    first_valid_index = auxiliary_series.first_valid_index()
    
    # ======= IV. Classify NaNs =======
    if first_valid_index is None:
        return auxiliary_series, 1.0, nan_indexes, [], []

    beginning_nans = nan_indexes[nan_indexes < first_valid_index]  
    middle_nans = nan_indexes[nan_indexes >= first_valid_index] 
    
    # ======= V. Identify infinite values =======
    infinite_indexes = auxiliary_series[np.isinf(auxiliary_series)].index
    
    # ======= VI. Calculate the proportion of error values =======
    total_values = len(auxiliary_series) - len(beginning_nans)
    total_error_values = len(middle_nans) + len(infinite_indexes)
    error_proportion = total_error_values / total_values if total_values > 0 else 0
    
    # ======= VII. Create a Clean Series =======
    clean_series = auxiliary_series.replace([np.inf, -np.inf], np.nan)  
    clean_series = clean_series.fillna(method="ffill") 
    
    return clean_series, error_proportion, beginning_nans.tolist(), middle_nans.tolist(), infinite_indexes.tolist()
    
#*____________________________________________________________________________________ #
def check_for_outliers(
    feature_series: pd.Series, 
    threshold: float = 3
) -> tuple:
    """
    This function checks for outliers in a given feature series using Z-scores.
    Parameters:
        - feature_series (pd.Series): The feature series to be checked for outliers.
        - threshold (float): The Z-score threshold for identifying outliers.
    
    Returns:
        - filtered_series (pd.Series): A series filtered from outliers.
        - outliers_df (pd.DataFrame): A DataFrame containing the indexes, values, and Z-scores of the outliers.
    """
    # ======= I. Check if standard deviation is zero =======
    std = feature_series.std()
    if std == 0:
        return pd.DataFrame(columns=["index", "value", "z-score", "threshold"]), pd.Series(dtype="float64")

    # ======= II. Compute Z-scores =======
    mean = feature_series.mean()
    z_scores = (feature_series - mean) / std

    # ======= III. Identify outliers =======
    outliers = feature_series[z_scores.abs() > threshold]

    # ======= IV. Create a DataFrame to store outliers =======
    threshold_value = threshold * std + mean
    outliers_df = pd.DataFrame({
        "index": outliers.index,
        "value": outliers.values.astype("float64"),
        "z-score": z_scores[outliers.index].astype("float64"),
        "threshold": threshold_value
    }).reset_index(drop=True)
    
    # ======= V. Create a series filtered from outliers =======
    filtered_series = feature_series.copy()
    filtered_series.loc[outliers.index] = np.nan
    
    return filtered_series, outliers_df
    
#*____________________________________________________________________________________ #
def check_for_scale(
    feature_series: pd.Series, 
    mean_tolerance: float = 0.01, 
    std_tolerance: float = 0.01, 
    range_tolerance: float = 0.1
) -> tuple:
    """
    This function checks for scale in a given feature series.
    Parameters:
        - feature_series (pd.Series): The feature series to be checked for scale.
        - mean_tolerance (float): The tolerance for the mean value.
        - std_tolerance (float): The tolerance for the standard deviation.
        - range_tolerance (float): The tolerance for the range of values.
    
    Returns:
        - auxiliary_series (pd.Series): The normalized series.
        - mean (float): The mean of the series.
        - std (float): The standard deviation of the series.
        - min_val (float): The minimum value of the series.
        - max_val (float): The maximum value of the series.
    """
    # ======= I. Check feature characteristics =======
    auxiliary_series = feature_series.copy()
    mean = auxiliary_series.mean()
    std = auxiliary_series.std()
    min_val = auxiliary_series.min()
    max_val = auxiliary_series.max()

    is_mean_near_zero = abs(mean) < mean_tolerance
    is_std_near_one = abs(std - 1) < std_tolerance
    is_values_in_range = auxiliary_series.between(-1 - range_tolerance, 1 + range_tolerance).all()

    # ======= II. Apply necessary normalization =======
    if not is_mean_near_zero:
        auxiliary_series -= mean
    if not is_std_near_one:
        auxiliary_series /= std
    if not is_values_in_range:
        auxiliary_series = (auxiliary_series - min_val) / (max_val - min_val)
        auxiliary_series = auxiliary_series * 2 - 1
    
    return auxiliary_series, mean, std, min_val, max_val

#*____________________________________________________________________________________ #
def check_for_stationarity(
    feature_series: pd.Series, 
    threshold: float = 0.05
) -> tuple:
    """
    This function checks for stationarity in a given feature series using the Augmented Dickey-Fuller (ADF) test and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.
    It returns a tuple indicating whether the series is stationary according to each test.
    Parameters:
        - features_series (pd.Series): The feature series to be checked for stationarity.
        - threshold (float): The significance level for the stationarity tests.
    
    Returns:
        - (tuple): A tuple containing two boolean values indicating whether the series is stationary according to the ADF and KPSS tests, respectively.
    """
    # ======= I. Perform Stationarity Tests =======
    adf_results = adfuller(feature_series)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_results = kpss(feature_series, regression="c", nlags="auto")
    
    # ======= II. Check p-values =======
    adf_p_value = adf_results[1]
    kpss_p_value = kpss_results[1]
    
    # ======= III. Determine Stationarity =======
    is_adf_stationary = adf_p_value < threshold
    is_kpss_stationary = kpss_p_value > threshold
    
    return is_adf_stationary, is_kpss_stationary