import sys
sys.path.append("../")
from Features import auxiliary as aux
from Features import common as cm

import pandas as pd
import numpy as np

#! ==================================================================================== #
#! =========================== Relationship Measures Features ========================= #
class cointegration_feature(cm.Feature):
    def __init__(
        self, 
        data: tuple | pd.DataFrame, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window_sizes": [5, 10, 30, 60, 120, 240],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self):
        if isinstance(self.data, pd.DataFrame):
            series_1 = self.data.iloc[:, 0]
            series_2 = self.data.iloc[:, 1]
            self.processed_data = (series_1, series_2)
        else:
            self.processed_data = self.data.copy()

        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        data: tuple,
        window: int,
    ):
        # ======== I. Initialize Outputs (Pre-allocate for performance) ========
        series_1 = data[0]
        series_2 = data[1]

        num_obs = len(series_1) - window
        beta_values = np.full(num_obs, np.nan)
        intercept_values = np.full(num_obs, np.nan)
        adf_p_values = np.full(num_obs, np.nan)
        kpss_p_values = np.full(num_obs, np.nan)
        residuals_values = np.full(num_obs, np.nan)
        
        # ======== II. Iterate Over Observations ========
        for i in range(num_obs):
            # II.1 Extract Time Windows
            series1_window = series_1.iloc[i : i + window]
            series2_window = series_2.iloc[i : i + window]
            
            # II.2 Perform Cointegration Test
            beta, intercept, adf_results, kpss_results, residuals = aux.cointegration_test(series_1=series1_window, series_2=series2_window)
            
            # II.3 Store Results
            beta_values[i] = beta
            intercept_values[i] = intercept
            adf_p_values[i] = adf_results[1]  
            kpss_p_values[i] = kpss_results[1]  
            residuals_values[i] = residuals[-1] 
        
        # ======== III. Convert to Series ========
        index = series_1.index[window:]
        
        beta_series = pd.Series(beta_values, index=index)
        intercept_series = pd.Series(intercept_values, index=index)
        adf_p_values_series = pd.Series(adf_p_values, index=index)
        kpss_p_values_series = pd.Series(kpss_p_values, index=index)
        residuals_series = pd.Series(residuals_values, index=index)
        
        # ======== IV. Change Names ========
        beta_series.name = f"beta_{window}"
        intercept_series.name = f"intercept_{window}"
        adf_p_values_series.name = f"ADF_pvalue_{window}"
        kpss_p_values_series.name = f"KPSS_pvalue_{window}"
        residuals_series.name = f"residuals_{window}"
        
        return beta_series, intercept_series, adf_p_values_series, kpss_p_values_series, residuals_series


#! ==================================================================================== #
#! ============================== Spread Series Features ============================== #
class OU_feature(cm.Feature):
    def __init__(
        self, 
        data: tuple | pd.DataFrame, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window_sizes": [5, 10, 30, 60, 120, 240],
                "residuals_weights": None,
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self):
        if isinstance(self.data, pd.DataFrame):
            series_1 = self.data.iloc[:, 0]
            series_2 = self.data.iloc[:, 1]
            self.processed_data = (series_1, series_2)
        else:
            self.processed_data = self.data.copy()

        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        data: tuple,
        window: int,
        residuals_weights: np.array,
    ):
        # ======== I. Initialize Outputs (Pre-allocate for performance) ========
        series_1 = data[0]
        series_2 = data[1]

        num_obs = len(series_1) - window
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
                _, _, _, _, residuals = aux.cointegration_test(series_1=series1_window, series_2=series2_window)
            else: 
                residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
            
            # II.3 Perform Ornstein-Uhlenbeck Estimation
            mu, theta, sigma, half_life = aux.ornstein_uhlenbeck_estimation(series=residuals)
            
            # II.4 Store Results
            mu_values[i] = mu
            theta_values[i] = theta
            sigma_values[i] = sigma
            half_life_values[i] = half_life
        
        # ======== III. Convert to Series ========
        index = series_1.index[window:]
        
        mu_series = pd.Series(mu_values, index=index)
        theta_series = pd.Series(theta_values, index=index)
        sigma_series = pd.Series(sigma_values, index=index)
        half_life_series = pd.Series(half_life_values, index=index)
        
        # ======== IV. Change Names ========
        mu_series.name = f"OU_mu_{window}"
        theta_series.name = f"OU_theta_{window}"
        sigma_series.name = f"OU_sigma_{window}"
        half_life_series.name = f"OU_half_life_{window}"
        
        return mu_series, theta_series, sigma_series, half_life_series

#*____________________________________________________________________________________ #
class OU_feature(cm.Feature):
    def __init__(
        self, 
        data: tuple | pd.DataFrame, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window_sizes": [5, 10, 30, 60, 120, 240],
                "residuals_weights": None,
                "smooth_coefficient": [0.1, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self):
        if isinstance(self.data, pd.DataFrame):
            series_1 = self.data.iloc[:, 0]
            series_2 = self.data.iloc[:, 1]
            self.processed_data = (series_1, series_2)
        else:
            self.processed_data = self.data.copy()

        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        data: tuple,
        window: int,
        residuals_weights: np.array,
        smooth_coefficient: float,
    ):
        # ======== I. Initialize Outputs (Pre-allocate for performance) ========
        series_1 = data[0]
        series_2 = data[1]

        num_obs = len(series_1) - window
        state_values = np.full(num_obs, np.nan)
        variance_values = np.full(num_obs, np.nan)
        
        # ======== II. Iterate Over Observations ========
        for i in range(num_obs):
            # II.1 Extract Time Windows
            series1_window = series_1.iloc[i : i + window]
            series2_window = series_2.iloc[i : i + window]
            
            # II.2 Extract residuals from cointegration test
            if residuals_weights is None:
                _, _, _, _, residuals = aux.cointegration_test(series_1=series1_window, series_2=series2_window)
            else: 
                residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
            
            # II.3 Perform Ornstein-Uhlenbeck Estimation
            filtered_states, variances = aux.kalmanOU_estimation(series=residuals, smooth_coefficient=smooth_coefficient)
            
            # II.4 Store Results
            state_values[i] = filtered_states[-1]
            variance_values[i] = variances[-1]
        
        # ======== III. Convert to Series ========
        index = series_1.index[window:]
        
        state_series = pd.Series(state_values, index=index)
        variance_series = pd.Series(variance_values, index=index)
        
        # ======== IV. Change Names ========
        state_series.name = f"KF_state_{window}"
        variance_series.name = f"KF_variance_{window}"
        
        return state_series, variance_series