from ..Measures import Filters as fil
from ..Measures import Codependence as cod
from ..Measures import Momentum as mom

from ..Features import common as com

import pandas as pd
import numpy as np
from typing import Union

#! ==================================================================================== #
#! =========================== Relationship Measures Features ========================= #
class Cointegration_feature(com.Feature):
    def __init__(
        self, 
        data: Union[tuple, pd.DataFrame], 
        name: str = "cointegration", 
        params: dict = None,  
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10],
                "lambda_smooth": [0.1, 0.2, 0.5],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
        )

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ) -> pd.DataFrame:
        """
        This method applies a smoothing filter to the series before calculating the features.
        Parameters:
            - smoothing_method (str): The smoothing method to be applied. Options are "ewma" or "average".
            - window_smooth (int): The window size for the smoothing method. It should a number of bars.
            - lambda_smooth (float): The lambda parameter for the ewma method. It should be in [0, 1].
            
        Returns:
            - processed_data (pd.DataFrame): A DataFrame containing the smoothed series.
        """
        # ======= I. Ensure Data is in DataFrame Format =======
        if isinstance(self.data, pd.DataFrame):
            self.processed_data = self.data.copy()
        else:
            self.processed_data = pd.DataFrame({"series_1": self.data[0], "series_2": self.data[1]})

        # ======== II. Apply Smoothing if Needed ========
        if smoothing_method is None:
            return self.processed_data
        
        else:
            smoothed_data = {}
            for col in self.processed_data.columns:
                series = self.processed_data[col]

                if smoothing_method == "ewma":
                    smoothed_series = fil.ewma_smoothing(price_series=series, window=window_smooth, ind_lambda=lambda_smooth)
                elif smoothing_method == "average":
                    smoothed_series = fil.average_smoothing(price_series=series, window=window_smooth)
                else:
                    raise ValueError(f"Smoothing method '{smoothing_method}' not recognized")

                smoothed_data[col] = smoothed_series

            self.processed_data = pd.DataFrame(smoothed_data)

        return self.processed_data

    #?____________________________________________________________________________________ #
    def get_feature(
        self, 
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.DataFrame:
        """
        This method computes the cointegration informations between the two time series.
        It uses the Engle-Granger two-step method to estimate the cointegration relationship and returns the beta, intercept, ADF p-value, KPSS p-value, and residuals.
        Parameters:
            - window (int): The window size for the rolling calculation of the cointegration test.
            
            - smoothing_method (str): see process_data.
            - window_smooth (int): see process_data.
            - lambda_smooth (float): see process_data.
        
        Returns:
            - features_df (pd.DataFrame): A DataFrame containing the cointegration features.
        """
        # ======= I. Process Data =======
        processed_data = self.process_data(smoothing_method, window_smooth, lambda_smooth)
        series_1 = processed_data["series_1"]
        series_2 = processed_data["series_2"]

        # ======= II. Ensure the window is not too large =======
        num_obs = len(series_1) - window
        if num_obs <= 0:
            raise ValueError(f"Window size {window} is too large for the given data length {len(series_1)}.")
        
        # ======= III. Initialize Output Arrays =======
        beta_values = np.full(num_obs, np.nan)
        intercept_values = np.full(num_obs, np.nan)
        adf_p_values = np.full(num_obs, np.nan)
        kpss_p_values = np.full(num_obs, np.nan)
        residuals_values = np.full(num_obs, np.nan)

        # ======== IV. Iterate Over Observations ========
        for i in range(num_obs):
            # IV.1 Extract Time Windows
            series1_window = series_1.iloc[i : i + window]
            series2_window = series_2.iloc[i : i + window]

            # IV.2 Perform Cointegration Test
            beta, intercept, adf_results, kpss_results, residuals = cod.get_cointegration(series_1=series1_window, series_2=series2_window)

            # IV.3 Store Results
            beta_values[i] = beta
            intercept_values[i] = intercept
            adf_p_values[i] = adf_results[1]  
            kpss_p_values[i] = kpss_results[1]  
            residuals_values[i] = residuals[-1]

        # ======== V. Create the Final DataFrame ========
        index = series_1.index[window:]
        features_df = pd.DataFrame({
            f"beta_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": beta_values,
            f"intercept_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": intercept_values,
            f"ADF_pvalue_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": adf_p_values,
            f"KPSS_pvalue_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": kpss_p_values,
            f"residuals_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": residuals_values,
        }, index=index)

        return features_df


#! ==================================================================================== #
#! ============================== Spread Series Features ============================== #
class OU_feature(com.Feature):
    def __init__(
        self, 
        data: Union[tuple, pd.DataFrame], 
        name: str = "OrnsteinUhlenbeck", 
        params: dict = None,  # Fixed: Changed from list to dict
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60],
                "residuals_weights": [None, [1, 0]],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10],
                "lambda_smooth": [0.1, 0.2, 0.5],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
        )

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ) -> pd.DataFrame:
        """
        This method applies a smoothing filter to the series before calculating the features.
        Parameters:
            - smoothing_method (str): The smoothing method to be applied. Options are "ewma" or "average".
            - window_smooth (int): The window size for the smoothing method. It should a number of bars.
            - lambda_smooth (float): The lambda parameter for the ewma method. It should be in [0, 1].
            
        Returns:
            - processed_data (pd.DataFrame): A DataFrame containing the smoothed series.
        """
        # ======= I. Ensure Data is in DataFrame Format =======
        if isinstance(self.data, pd.DataFrame):
            self.processed_data = self.data.copy()
        else:
            self.processed_data = pd.DataFrame({"series_1": self.data[0], "series_2": self.data[1]})

        # ======== II. Apply Smoothing if Needed ========
        if smoothing_method is None:
            return self.processed_data
        
        else:
            smoothed_data = {}
            for col in self.processed_data.columns:
                series = self.processed_data[col]

                if smoothing_method == "ewma":
                    smoothed_series = fil.ewma_smoothing(price_series=series, window=window_smooth, ind_lambda=lambda_smooth)
                elif smoothing_method == "average":
                    smoothed_series = fil.average_smoothing(price_series=series, window=window_smooth)
                else:
                    raise ValueError(f"Smoothing method '{smoothing_method}' not recognized")

                smoothed_data[col] = smoothed_series

            self.processed_data = pd.DataFrame(smoothed_data)

        return self.processed_data

    #?____________________________________________________________________________________ #
    def get_feature(
        self, 
        window: int,
        residuals_weights: list,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        This method estimates the Ornstein-Uhlenbeck parameters for the spread between two time series. The Spread is defined as the difference between the two time series.
        It can be computed using the cointegration residuals or using fixed weights. It returns the mu, theta, sigma, and half-life of the Ornstein-Uhlenbeck process.
        Parameters:
            - window (int): The window size for the rolling calculation of the Ornstein-Uhlenbeck parameters.
            - residuals_weights (list): The weights to be used to compute the spread. If None, the cointegration residuals are used.
            
            - smoothing_method (str): see process_data.
            - window_smooth (int): see process_data.
            - lambda_smooth (float): see process_data.
        
        Returns: 
            - features_df (pd.DataFrame): A DataFrame containing the Ornstein-Uhlenbeck parameters.
        """
        # ======= I. Process Data =======
        processed_data = self.process_data(smoothing_method, window_smooth, lambda_smooth)
        series_1 = processed_data["series_1"]
        series_2 = processed_data["series_2"]
        
        # ======= II. Ensure the window is not too large =======
        num_obs = len(series_1) - window
        if num_obs <= 0:
            raise ValueError(f"Window size {window} is too large for the given data length {len(series_1)}.")
        
        # ======= III. Initialize Output Arrays =======
        mu_values = np.full(num_obs, np.nan)
        theta_values = np.full(num_obs, np.nan)
        sigma_values = np.full(num_obs, np.nan)
        half_life_values = np.full(num_obs, np.nan)
        
        # ======= IV. Iterate Over Observations ========
        for i in range(num_obs):
            # IV.1 Extract Time Windows
            series1_window = series_1.iloc[i : i + window]
            series2_window = series_2.iloc[i : i + window]
            
            # IV.2 Extract residuals from cointegration test
            if residuals_weights is None:
                _, _, _, _, residuals = cod.get_cointegration(series_1=series1_window, series_2=series2_window)
            else: 
                residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
            
            # IV.3 Perform Ornstein-Uhlenbeck Estimation
            mu, theta, sigma, half_life = mom.get_OU_estimation(series=residuals)
            
            # IV.4 Store Results
            mu_values[i] = mu
            theta_values[i] = theta
            sigma_values[i] = sigma
            half_life_values[i] = half_life
        
        # ======= V. Create the Final DataFrame ========
        index = series_1.index[window:]
        features_df = pd.DataFrame({
            f"OU_mu_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": mu_values,
            f"OU_theta_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": theta_values,
            f"OU_sigma_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": sigma_values,
            f"OU_half_life_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": half_life_values,
        }, index=index)
        
        return features_df


#*____________________________________________________________________________________ #
class kalmanOU_feature(com.Feature):
    def __init__(
        self, 
        data: Union[tuple, pd.DataFrame], 
        name: str = "kalmanOU", 
        params: dict = None,  # Fixed: Changed from list to dict
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60],
                "residuals_weights": [None, [1, 0]],
                "smooth_coefficient": [0.2, 0.5, 0.8],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10],
                "lambda_smooth": [0.1, 0.2, 0.5],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
        )

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ) -> pd.DataFrame:
        """
        This method applies a smoothing filter to the series before calculating the features.
        Parameters:
            - smoothing_method (str): The smoothing method to be applied. Options are "ewma" or "average".
            - window_smooth (int): The window size for the smoothing method. It should a number of bars.
            - lambda_smooth (float): The lambda parameter for the ewma method. It should be in [0, 1].
            
        Returns:
            - processed_data (pd.DataFrame): A DataFrame containing the smoothed series.
        """
        # ======= I. Ensure Data is in DataFrame Format =======
        if isinstance(self.data, pd.DataFrame):
            self.processed_data = self.data.copy()
        else:
            self.processed_data = pd.DataFrame({"series_1": self.data[0], "series_2": self.data[1]})

        # ======== II. Apply Smoothing if Needed ========
        if smoothing_method is None:
            return self.processed_data
        
        else:
            smoothed_data = {}
            for col in self.processed_data.columns:
                series = self.processed_data[col]

                if smoothing_method == "ewma":
                    smoothed_series = fil.ewma_smoothing(price_series=series, window=window_smooth, ind_lambda=lambda_smooth)
                elif smoothing_method == "average":
                    smoothed_series = fil.average_smoothing(price_series=series, window=window_smooth)
                else:
                    raise ValueError(f"Smoothing method '{smoothing_method}' not recognized")

                smoothed_data[col] = smoothed_series

            self.processed_data = pd.DataFrame(smoothed_data)

        return self.processed_data

    #?____________________________________________________________________________________ #
    def get_feature(
        self, 
        window: int,
        residuals_weights: list,
        smooth_coefficient: float,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        This method estimates the filtered States and Variances of an Ornstein-Uhlenbeck process using the Kalman filter.
        It can be computed using the cointegration residuals or using fixed weights. It returns the filtered states and variances of the Ornstein-Uhlenbeck process.
        Parameters:
            - window (int): The window size for the rolling calculation of the Ornstein-Uhlenbeck parameters.
            - residuals_weights (list): The weights to be used to compute the spread. If None, the cointegration residuals are used.
            - smooth_coefficient (float): The smoothing coefficient for the Kalman filter. It should be in [0, 1].
            
            - smoothing_method (str): see process_data.
            - window_smooth (int): see process_data.
            - lambda_smooth (float): see process_data.
        """
        # ======== I. Process Data ========
        processed_data = self.process_data(smoothing_method, window_smooth, lambda_smooth)
        series_1 = processed_data["series_1"]
        series_2 = processed_data["series_2"]
        
        # ======== II. Ensure the window is not too large ========
        num_obs = len(series_1) - window
        if num_obs <= 0:
            raise ValueError(f"Window size {window} is too large for the given data length {len(series_1)}.")
        
        # ======== III. Initialize Output Arrays ========
        state_values = np.full(num_obs, np.nan)
        variance_values = np.full(num_obs, np.nan)
        
        # ======= IV. Iterate Over Observations ========
        for i in range(num_obs):
            # IV.1 Extract Time Windows
            series1_window = series_1.iloc[i : i + window]
            series2_window = series_2.iloc[i : i + window]
            
            # IV.2 Extract residuals from cointegration test
            if residuals_weights is None:
                _, _, _, _, residuals = cod.get_cointegration(series_1=series1_window, series_2=series2_window)
            else: 
                residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
            
            # IV.3 Perform Ornstein-Uhlenbeck Estimation
            filtered_states, variances = fil.kalmanOU_smoothing(series=residuals, smooth_coefficient=smooth_coefficient)
            
            # IV.4 Store Results
            state_values[i] = filtered_states[-1]
            variance_values[i] = variances[-1]
        
        # ======== V. Create the Final DataFrame ========
        index = series_1.index[window:]
        features_df = pd.DataFrame({
            f"KF_state_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": state_values,
            f"KF_variance_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": variance_values,
        }, index=index)
        
        return features_df

