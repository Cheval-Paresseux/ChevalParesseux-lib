from ..Measures import Filters as fil
from ..Measures import Codependence as cod
from ..Measures import Momentum as mom

from ..Features import common as com

import pandas as pd
import numpy as np
from typing import Union

#! ==================================================================================== #
#! =========================== Relationship Measures Features ========================= #
class CointegrationFeature(com.Feature):
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
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
        )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if isinstance(self.data, pd.DataFrame):
            self.processed_data = self.data.copy()
        else:
            self.processed_data = pd.DataFrame({"series_1": self.data[0], "series_2": self.data[1]})

        # Apply smoothing if needed
        if smoothing_method is not None:
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
    ):
        processed_data = self.process_data(smoothing_method, window_smooth, lambda_smooth)
        series_1 = processed_data["series_1"]
        series_2 = processed_data["series_2"]

        num_obs = len(series_1) - window
        if num_obs <= 0:
            raise ValueError(f"Window size {window} is too large for the given data length {len(series_1)}.")
        
        beta_values = np.full(num_obs, np.nan)
        intercept_values = np.full(num_obs, np.nan)
        adf_p_values = np.full(num_obs, np.nan)
        kpss_p_values = np.full(num_obs, np.nan)
        residuals_values = np.full(num_obs, np.nan)

        # ======== Iterate Over Observations ========
        for i in range(num_obs):
            # Extract Time Windows
            series1_window = series_1.iloc[i : i + window]
            series2_window = series_2.iloc[i : i + window]

            # Perform Cointegration Test
            beta, intercept, adf_results, kpss_results, residuals = cod.get_cointegration(series_1=series1_window, series_2=series2_window)

            # Store Results
            beta_values[i] = beta
            intercept_values[i] = intercept
            adf_p_values[i] = adf_results[1]  
            kpss_p_values[i] = kpss_results[1]  
            residuals_values[i] = residuals[-1]

        # Convert to Pandas Series with Proper Indexing
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
                "window": [5, 10, 30, 60, 120, 240],
                "residuals_weights": [None],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
        )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if isinstance(self.data, pd.DataFrame):
            self.processed_data = self.data.copy()
        else:
            self.processed_data = pd.DataFrame({"series_1": self.data[0], "series_2": self.data[1]})

        # Apply smoothing if needed
        if smoothing_method is not None:
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
        residuals_weights: np.array,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        processed_data = self.process_data(smoothing_method, window_smooth, lambda_smooth)
        series_1 = processed_data["series_1"]
        series_2 = processed_data["series_2"]
        
        num_obs = len(series_1) - window
        if num_obs <= 0:
            raise ValueError(f"Window size {window} is too large for the given data length {len(series_1)}.")
        
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
                _, _, _, _, residuals = cod.get_cointegration(series_1=series1_window, series_2=series2_window)
            else: 
                residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
            
            # II.3 Perform Ornstein-Uhlenbeck Estimation
            mu, theta, sigma, half_life = mom.get_OU_estimation(series=residuals)
            
            # II.4 Store Results
            mu_values[i] = mu
            theta_values[i] = theta
            sigma_values[i] = sigma
            half_life_values[i] = half_life
        
        # ======== III. Convert to Series ========
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
                "window": [5, 10, 30, 60, 120, 240],
                "residuals_weights": [None],
                "smooth_coefficient": [0.1, 0.3, 0.5, 0.7, 0.9],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
        )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if isinstance(self.data, pd.DataFrame):
            self.processed_data = self.data.copy()
        else:
            self.processed_data = pd.DataFrame({"series_1": self.data[0], "series_2": self.data[1]})

        # Apply smoothing if needed
        if smoothing_method is not None:
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
        residuals_weights: np.array,
        smooth_coefficient: float,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        processed_data = self.process_data(smoothing_method, window_smooth, lambda_smooth)
        series_1 = processed_data["series_1"]
        series_2 = processed_data["series_2"]
        
        num_obs = len(series_1) - window
        if num_obs <= 0:
            raise ValueError(f"Window size {window} is too large for the given data length {len(series_1)}.")
        
        state_values = np.full(num_obs, np.nan)
        variance_values = np.full(num_obs, np.nan)
        
        # ======== II. Iterate Over Observations ========
        for i in range(num_obs):
            # II.1 Extract Time Windows
            series1_window = series_1.iloc[i : i + window]
            series2_window = series_2.iloc[i : i + window]
            
            # II.2 Extract residuals from cointegration test
            if residuals_weights is None:
                _, _, _, _, residuals = cod.get_cointegration(series_1=series1_window, series_2=series2_window)
            else: 
                residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
            
            # II.3 Perform Ornstein-Uhlenbeck Estimation
            filtered_states, variances = fil.kalmanOU_smoothing(series=residuals, smooth_coefficient=smooth_coefficient)
            
            # II.4 Store Results
            state_values[i] = filtered_states[-1]
            variance_values[i] = variances[-1]
        
        # ======== III. Convert to Series ========
        index = series_1.index[window:]
        features_df = pd.DataFrame({
            f"KF_state_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": state_values,
            f"KF_variance_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": variance_values,
        }, index=index)
        
        return features_df

