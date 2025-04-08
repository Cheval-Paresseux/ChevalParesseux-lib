import os

os.environ["MKL_NUM_THREADS"] = "1"
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
import pandas as pd
import numpy as np

#! ==================================================================================== #
#! ================================= Main Function ==================================== #
def check_feature(
    feature_series: pd.Series,
    stationarity_threshold: float = 0.05, 
    outliers_threshold: float = 5, 
    mean_tolerance: float = 0.01, 
    std_tolerance: float = 0.01, 
):
    """
    This function checks a given feature series for various characteristics and store the results in a DataFrame.
    Those values should be used to determine if the feature is suitable for training.
    If some transformations are applied, you should use the transformation values for the testing data.
    Parameters:
        - feature_series (pd.Series): The feature series to be checked.
        - stationarity_threshold (float): The threshold for stationarity tests.
        - outliers_threshold (float): The threshold for outlier detection.
        - mean_tolerance (float): The tolerance for the mean value.
        - std_tolerance (float): The tolerance for the standard deviation.
        - range_tolerance (float): The tolerance for the range of values.
    
    Returns:
        - scaled_series (pd.Series): The normalized series.
        - results_df (pd.DataFrame): A DataFrame containing the results of the checks.
    """
    # ======= I. Check for error values =======
    clean_series, error_proportion, beginning_nans, middle_nans, infinite_indexes = check_for_error_values(feature_series=feature_series)
    
    # ======= II. Check for outliers =======
    filtered_series, outliers_df, threshold_value = check_for_outliers(feature_series=clean_series,  threshold=outliers_threshold)
    
    # ======= III. Check for scale =======
    scaled_series, mean, std = check_for_scale(feature_series=filtered_series, mean_tolerance=mean_tolerance, std_tolerance=std_tolerance)
    
    # ======= IV. Check for stationarity =======
    dropped_series = scaled_series.dropna()
    is_adf_stationary, is_kpss_stationary = check_for_stationarity(feature_series=dropped_series, threshold=stationarity_threshold)

    outliers_proportion = len(outliers_df) / len(dropped_series) if len(dropped_series) > 0 else 0
    # ======= V. Store results inside a DataFrame =======
    results_df = pd.DataFrame({
        "feature_name": feature_series.name,
        "error_proportion": error_proportion,
        "beginning_nans": len(beginning_nans),
        "middle_nans": len(middle_nans),
        "infinite_indexes": len(infinite_indexes),
        "outliers_count": len(outliers_df),
        "outliers_proportion": outliers_proportion,
        "outliers_threshold": threshold_value,
        "mean": mean,
        "std": std,
        "is_adf_stationary": is_adf_stationary,
        "is_kpss_stationary": is_kpss_stationary
    }, index=[0])
    
    return scaled_series, results_df


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
    clean_series = clean_series.ffill()
    
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
    series = feature_series.copy()
    std = series.std()

    # ======= II. Compute Z-scores =======
    mean = series.mean()
    z_scores = (series - mean) / std

    # ======= III. Identify outliers =======
    outliers = series[z_scores.abs() > threshold]

    # ======= IV. Create a DataFrame to store outliers =======
    threshold_value = threshold * std + mean
    outliers_df = pd.DataFrame({
        "index": outliers.index,
        "value": outliers.values.astype("float64"),
        "z-score": z_scores[outliers.index].astype("float64"),
    }).reset_index(drop=True)
    
    # ======= V. Create a series filtered from outliers =======
    filtered_series = series.copy()
    filtered_series.loc[outliers.index] = np.nan
    
    return filtered_series, outliers_df, threshold_value
    
#*____________________________________________________________________________________ #
def check_for_scale(
    feature_series: pd.Series, 
    mean_tolerance: float = 0.01, 
    std_tolerance: float = 0.01, 
) -> tuple:
    """
    This function checks for scale in a given feature series.
    Parameters:
        - feature_series (pd.Series): The feature series to be checked for scale.
        - mean_tolerance (float): The tolerance for the mean value.
        - std_tolerance (float): The tolerance for the standard deviation.
    
    Returns:
        - auxiliary_series (pd.Series): The normalized series.
        - mean (float): The mean of the series.
        - std (float): The standard deviation of the series.
    """
    # ======= I. Check feature characteristics =======
    auxiliary_series = feature_series.copy()
    mean = auxiliary_series.mean()
    std = auxiliary_series.std()

    is_mean_near_zero = abs(mean) < mean_tolerance
    is_std_near_one = abs(std - 1) < std_tolerance

    # ======= II. Apply necessary normalization =======
    if not is_mean_near_zero:
        auxiliary_series -= mean
    if not is_std_near_one:
        auxiliary_series /= std
    
    return auxiliary_series, mean, std

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