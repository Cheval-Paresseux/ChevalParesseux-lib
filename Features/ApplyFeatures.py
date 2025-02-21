import os
import sys

sys.path.append(os.path.abspath("../"))
import Smoothing.NoLook_Filters as filters
import Features.SingleTS as single_ts
import Features.PredictionsTS as predictions_ts
import Features.DualTS as dual_ts

import pandas as pd
from tqdm import tqdm


def singleTS_features(data_df: pd.DataFrame, features_params: dict):
    """
    Compute the features for a single time series.
    
    Args:
        data_df (pd.DataFrame): The dataframe containing the data to decompose (each column representing the time series for an asset).
        features_params (dict): The dictionary containing the parameters for the features computation.
        ---------------------
        Example of features_params:
        features_params = {
            "filter_windows": [5, 10, 20, 50],
            "filter_series": ["MA", "EWMA"],
            "rolling_windows": [5, 10, 20, 50],
            "ind_lambda": 0.6,
            "wav_family": ["db4"],
            "decomposition_level": 2,
            "hurst_power": [2, 3, 4]
        }
        ---------------------
    
    Returns:
        featured_dfs_list (list): The list of individual dataframes with the features.
    """
    # ======= 0. Auxiliary Function =======
    def decompose_data(data_df: pd.DataFrame):
        """
        This method decomposes the data into individual dataframes for each asset in order to apply the labels and compute the features.
        It saves the individual dataframes in the instance variable individual_dfs_list which is a list of dataframes.

        Args:
            data_df (pd.DataFrame): The dataframe containing the data to decompose (optional because stored at class initialization).

        Returns:
            individual_dfs_list (list): The list of individual dataframes.
        """
        # ======= I. Initialize input and output =======
        individual_dfs_list = []

        # ======= II. Decompose the data =======
        for col in data_df.columns:
            asset_df = pd.DataFrame(data_df[col])
            individual_dfs_list.append(asset_df)

        return individual_dfs_list

    # ======= I. Initialize the input and output =======
    individual_dfs_list = decompose_data(data_df=data_df)
    featured_dfs_list = []

    # ======= II. Apply the features =======
    for asset_df in tqdm(individual_dfs_list):
        # II.1. Initialize the dataframe with the asset price
        asset_name = asset_df.columns[0]
        features_dict = {}
        series_list = {}

        # ------- II.2 Store the different series on which we apply the features -------
        raw_price_series = asset_df[asset_name]
        series_list["RAW"] = raw_price_series

        for filtering_window in features_params["filter_windows"]:
            if "MA" in features_params["filter_series"]:
                ma_price_series = filters.moving_average(price_series=raw_price_series, window=filtering_window)
                series_list[f"MA{filtering_window}"] = ma_price_series
                
            if "EWMA" in features_params["filter_series"]:
                ewma_price_series = filters.exponential_weighted_moving_average(price_series=raw_price_series, window=filtering_window, ind_lambda=0.6)
                series_list[f"EWMA{filtering_window}"] = ewma_price_series

        # ------- II.3 Compute the features for each series -------
        for series_name, price_series in series_list.items():
            print(f"Computing features for {series_name}")

            # II.3.i Extract the features parameters
            wav_family = features_params["wav_family"]
            decomposition_level = features_params["decomposition_level"]
            ind_lambda = features_params["ind_lambda"]
            
            # II.3.ii Compute the features related to the rolling windows and store them in the dictionary
            for rolling_window in features_params["rolling_windows"]:
                # ------- Compute the features -------
                min, max = single_ts.minMax_features(price_series=price_series, window=rolling_window)
                average, ewma = single_ts.smoothing_features(price_series=price_series, window=rolling_window, ind_lambda=ind_lambda)
                vol = single_ts.volatility_features(price_series=price_series, window=rolling_window)
                momentum, Z_momentum = single_ts.momentum_features(price_series=price_series, window=rolling_window)
                trend, tstat = single_ts.linear_tempReg_features(price_series=price_series, regression_window=rolling_window,)
                nonLin_trend, nonLin_acceleration, nonLin_tstat = single_ts.nonlinear_tempReg_features(price_series=price_series, regression_window=rolling_window)
                wavelet_tuple = single_ts.wavelets_features(price_series=price_series, wavelet_window=rolling_window, wav_family=wav_family, decomposition_level=decomposition_level)
                shannon, plugin, lempel_ziv, kontoyiannis = single_ts.entropy_features(price_series=price_series, window=rolling_window)
                
                # ------- Store the features in the dictionary -------
                features_dict[f"{series_name}_rolling_min_{rolling_window}"] = min
                features_dict[f"{series_name}_rolling_max_{rolling_window}"] = max
                features_dict[f"{series_name}_rolling_average_{rolling_window}"] = average
                features_dict[f"{series_name}_rolling_ewma_{rolling_window}"] = ewma
                features_dict[f"{series_name}_rolling_vol_{rolling_window}"] = vol
                features_dict[f"{series_name}_rolling_momentum_{rolling_window}"] = momentum
                features_dict[f"{series_name}_rolling_Z_momentum_{rolling_window}"] = Z_momentum
                features_dict[f"{series_name}_rolling_trend_{rolling_window}"] = trend
                features_dict[f"{series_name}_rolling_tstat_{rolling_window}"] = tstat
                features_dict[f"{series_name}_rolling_nonLin_trend_{rolling_window}"] = nonLin_trend
                features_dict[f"{series_name}_rolling_nonLin_acceleration_{rolling_window}"] = nonLin_acceleration
                features_dict[f"{series_name}_rolling_nonLin_tstat_{rolling_window}"] = nonLin_tstat
                features_dict[f"{series_name}_rolling_shannon_{rolling_window}"] = shannon
                features_dict[f"{series_name}_rolling_plugin_{rolling_window}"] = plugin
                features_dict[f"{series_name}_rolling_lempel_ziv_{rolling_window}"] = lempel_ziv
                features_dict[f"{series_name}_rolling_kontoyiannis_{rolling_window}"] = kontoyiannis
                for wavelet in wavelet_tuple:
                    features_dict[f"{series_name}_rolling_wavelet_{wavelet.name}_{rolling_window}"] = wavelet

            # II.3.iii Compute the features related to the Hurst exponent
            for power in features_params["hurst_power"]:
                # ------- Compute the features -------
                hurst_exponent, hurst_tstat, hurst_pvalue = single_ts.hurst_exponent_features(price_series=price_series, power=power)
                
                # ------- Store the features in the dictionary -------
                features_dict[f"{series_name}_rolling_hurst_exponent_{power}"] = hurst_exponent
                features_dict[f"{series_name}_rolling_hurst_tstat_{power}"] = hurst_tstat
                features_dict[f"{series_name}_rolling_hurst_pvalue_{power}"] = hurst_pvalue

        # II.4. Drop NaN values and save the featured dataframe
        features_df = pd.DataFrame(features_dict)
        asset_df = pd.concat([asset_df, features_df], axis=1)
        asset_df.dropna(inplace=True)
        
        featured_dfs_list.append(asset_df)

    return featured_dfs_list

# -----------------------------------------------------------------------------
def predictionsTS_features(predictions_df: pd.DataFrame, features_params: dict):
    """
    Computes the features for the predictions time series.
    
    Args:
        predictions_df (pd.DataFrame): The dataframe containing the predictions to decompose (each column representing the predictions for an asset).
        features_params (dict): The dictionary containing the parameters for the features computation.
        ---------------------
        Example of features_params:
        features_params = {
            "rolling_windows": [5, 10, 20, 50]
        }
        ---------------------
    
    Returns:
        asset_features_df (pd.DataFrame): The dataframe containing the features for the predictions time series.
    """
    # ======= I. Initialize the input and output =======
    features_dict = {}
    predictions_series = predictions_df["predictions"]
    
    # ======= II. Compute the features =======
    for rolling_window in features_params["rolling_windows"]:
        # ------- Compute the features -------
        average = predictions_ts.average_predictions_features(predictions_series=predictions_series, window=rolling_window)
        volatility = predictions_ts.volatility_predictions_features(predictions_series=predictions_series, window=rolling_window)
        changes = predictions_ts.predictions_changes_features(predictions_series=predictions_series, window=rolling_window)
        shannon, plugin, lempel_ziv, kontoyiannis = predictions_ts.entropy_predictions_features(predictions_series=predictions_series, window=rolling_window)

        # ------- Store the features in the dictionary -------
        features_dict[f"rolling_avg_predictions_{rolling_window}"] = average
        features_dict[f"rolling_predictions_vol_{rolling_window}"] = volatility
        features_dict[f"rolling_predictions_changes_{rolling_window}"] = changes
        features_dict[f"rolling_predictions_shannon_{rolling_window}"] = shannon
        features_dict[f"rolling_predictions_plugin_{rolling_window}"] = plugin
        features_dict[f"rolling_predictions_lempel_ziv_{rolling_window}"] = lempel_ziv
        features_dict[f"rolling_predictions_kontoyiannis_{rolling_window}"] = kontoyiannis
    
    # ======= III. Drop NaN values and save the featured dataframe =======
    features_df = pd.DataFrame(features_dict)
    asset_features_df = pd.concat([predictions_df, features_df], axis=1)
    asset_features_df.dropna(inplace=True)
    
    return asset_features_df

# -----------------------------------------------------------------------------
def dualTS_features(prices_series_df: pd.DataFrame, features_params: dict):
    """
    Computes the features for the dual time series.
    
    Args:
        prices_series_df (pd.DataFrame): The dataframe containing the prices to decompose (each column representing the prices for an asset).
        features_params (dict): The dictionary containing the parameters for the features computation.
        ---------------------
        Example of features_params:
        features_params = {
            "rolling_windows": [5, 10, 20, 50],
            "smooth_coefficient": 0.5,
            "residuals_weight": 0.5
        }
        ---------------------
    
    Returns:
        asset_features_df (pd.DataFrame): The dataframe containing the features for the dual time series.
    """
    # ======= I. Initialize the input and output =======
    series_df = prices_series_df.copy()
    features_dict = {}
    series_1 = prices_series_df.columns[0]
    series_2 = prices_series_df.columns[1]
    
    smooth_coefficient = features_params["smooth_coefficient"]
    residuals_weights = features_params["residuals_weight"]
    
    # ======= II. Compute the features =======
    for rolling_window in features_params["rolling_windows"]:
        # ------- Compute the features -------
        beta, intercept, adf_p_values, kpss_p_values, residuals = dual_ts.cointegration_features(series_1=series_1, series_2=series_2, window=rolling_window)
        mu, theta, sigma, half_life = dual_ts.ornstein_uhlenbeck_features(series_1=series_1, series_2=series_2, window=rolling_window, residuals_weights=residuals_weights)
        kf_state, kf_variance = dual_ts.kalmanOU_features(series_1=series_1, series_2=series_2, window=rolling_window, smooth_coefficient=smooth_coefficient, residuals_weights=residuals_weights)
        
        # ------- Store the features in the dictionary -------
        features_dict[f"rolling_cointegration_beta_{rolling_window}"] = beta
        features_dict[f"rolling_cointegration_intercept_{rolling_window}"] = intercept
        features_dict[f"rolling_cointegration_adf_p_values_{rolling_window}"] = adf_p_values
        features_dict[f"rolling_cointegration_kpss_p_values_{rolling_window}"] = kpss_p_values
        features_dict[f"rolling_ornstein_uhlenbeck_mu_{rolling_window}"] = mu
        features_dict[f"rolling_ornstein_uhlenbeck_theta_{rolling_window}"] = theta
        features_dict[f"rolling_ornstein_uhlenbeck_sigma_{rolling_window}"] = sigma
        features_dict[f"rolling_ornstein_uhlenbeck_half_life_{rolling_window}"] = half_life
        features_dict[f"rolling_kalmanOU_state_{rolling_window}"] = kf_state
        features_dict[f"rolling_kalmanOU_variance_{rolling_window}"] = kf_variance
    
    # ======= III. Drop NaN values and save the featured dataframe =======
    features_df = pd.DataFrame(features_dict)
    asset_features_df = pd.concat([series_df, features_df], axis=1)
    asset_features_df.dropna(inplace=True)
    
    return asset_features_df
