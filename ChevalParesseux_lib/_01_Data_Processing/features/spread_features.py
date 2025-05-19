from ..features import common as com
from ... import utils

import pandas as pd
import numpy as np
from typing import Union, Self



#! ==================================================================================== #
#! =========================== Relationship Measures Features ========================= #
class Cointegration_feature(com.Feature):
    """
    Rolling Cointegration Feature Extraction

    This class computes the cointegration statistics between two time series using the Engle-Granger two-step method, 
    with optional pre-smoothing filters.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving cointegration features over a rolling window
    """
    def __init__(
        self, 
        name: str = "cointegration", 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the Cointegration_feature object.

        Parameters:
            - name (str): Feature name.
            - n_jobs (int): Number of parallel jobs to use.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
        )

    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        window: list = [5, 10, 30, 60], 
        smoothing_method: list = [None, "ewma", "average"], 
        window_smooth: list = [5, 10], 
        lambda_smooth: list = [0.1, 0.2, 0.5]
    ) -> Self:
        """
        Sets the parameter grid for cointegration feature extraction.

        Parameters:
            - window (list): Rolling window sizes for cointegration tests.
            - smoothing_method (list): Smoothing method to apply before testing.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA smoothing.
        """
        self.params = {
            "window": window,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }
        
        return self

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: Union[tuple, pd.DataFrame],
    ) -> tuple:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
        
        Returns:
            - processed_data (tuple(pd.Series)): The two series to be used for cointegration testing.
        """
        # ======= I. Extract Series =======
        if isinstance(data, pd.DataFrame):
            nb_series = data.shape[1]
            if nb_series != 2:
                raise ValueError(f"DataFrame must have exactly 2 columns, but got {nb_series}.")
            
            series_1 = data.iloc[:, 0]
            series_2 = data.iloc[:, 1]
        
        elif isinstance(data, tuple) and len(data) == 2:
            series_1 = data[0]
            series_2 = data[1]
        else:
            raise ValueError("Data must be either a tuple of two series or a DataFrame with two columns.")
        
        # ======= II. Ensure Series have the same indexation =======
        series_df = pd.DataFrame({"series_1": series_1, "series_2": series_2})
        series_df = series_df.dropna()
        series_1 = series_df["series_1"]
        series_2 = series_df["series_2"]
        
        # ======= III. Return Processed Data =======    
        processed_data = (series_1, series_2)

        return processed_data

    #?____________________________________________________________________________________ #
    def get_feature(
        self, 
        data: Union[tuple, pd.DataFrame],
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.DataFrame:
        """
        Computes rolling cointegration statistics between two series.

        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
            - window (int): Rolling window size for cointegration testing.
            - smoothing_method (str): See process_data.
            - window_smooth (int): See process_data.
            - lambda_smooth (float): See process_data.

        Returns:
            - features_df (pd.DataFrame): Contains beta, intercept, ADF p-value, KPSS p-value, and residuals.
        """
        # ======= I. Process Data =======
        processed_data = self.process_data(data=data)
        series_1 = processed_data[0]
        series_2 = processed_data[1]

        # ======= II. Apply Smoothing if Needed =======
        if smoothing_method is not None:
            series_1 = self.smooth_data(data=series_1, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_2 = self.smooth_data(data=series_2, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)

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
            beta, intercept, adf_results, kpss_results, residuals = utils.get_cointegration(
                series_1=series1_window, 
                series_2=series2_window
            )

            # IV.3 Store Results
            beta_values[i] = beta
            intercept_values[i] = intercept
            adf_p_values[i] = adf_results[1]  
            kpss_p_values[i] = kpss_results[1]  
            if not residuals.empty:
                residuals_values[i] = residuals.iloc[-1]
            else:
                residuals_values[i] = np.nan

        # ======== V. Create the Final DataFrame ========
        index = series_1.index[window:]
        features_df = pd.DataFrame({
            f"{self.name}_beta_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": beta_values,
            f"{self.name}_intercept_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": intercept_values,
            f"{self.name}_ADF_pvalue_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": adf_p_values,
            f"{self.name}_KPSS_pvalue_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": kpss_p_values,
            f"{self.name}_residuals_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": residuals_values,
        }, index=index)

        return features_df



#! ==================================================================================== #
#! ============================== Spread Series Features ============================== #
class OU_feature(com.Feature):
    """
    Rolling Ornstein-Uhlenbeck Process Feature Extraction

    This class estimates Ornstein-Uhlenbeck (OU) parameters over a rolling window.
    It includes the long-term mean (mu), speed of reversion (theta), volatility (sigma), and the half-life of mean reversion.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving Ornstein-Uhlenbeck estimation features over a rolling window
    """
    def __init__(
        self, 
        name: str = "OrnsteinUhlenbeck", 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the OU_feature object with a pair of input series.

        Parameters:
            - name (str): Feature name.
            - n_jobs (int): Number of parallel jobs to use.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
        )
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        window: list = [5, 10, 30, 60],
        residuals_weights: list = [None, [1, 0]],
        smoothing_method: list = [None, "ewma", "average"],
        window_smooth: list = [5, 10],
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ) -> Self:
        """
        Sets the parameter grid for OU feature extraction.

        Parameters:
            - window (list): Rolling window sizes for OU parameter estimation.
            - residuals_weights (list): List of residual computation modes. If None, use cointegration residuals; otherwise, use fixed [weight, intercept].
            - smoothing_method (list): Smoothing method to apply before estimation.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA smoothing.
        """
        self.params = {
            "window": window,
            "residuals_weights": residuals_weights,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }
        
        return self

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: Union[tuple, pd.DataFrame],
    ) -> tuple:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
        
        Returns:
            - processed_data (tuple(pd.Series)): The two series to be used for cointegration testing.
        """
        # ======= I. Extract Series =======
        if isinstance(data, pd.DataFrame):
            nb_series = data.shape[1]
            if nb_series != 2:
                raise ValueError(f"DataFrame must have exactly 2 columns, but got {nb_series}.")
            
            series_1 = data.iloc[:, 0]
            series_2 = data.iloc[:, 1]
        
        elif isinstance(data, tuple) and len(data) == 2:
            series_1 = data[0]
            series_2 = data[1]
        else:
            raise ValueError("Data must be either a tuple of two series or a DataFrame with two columns.")
        
        # ======= II. Ensure Series have the same indexation =======
        series_df = pd.DataFrame({"series_1": series_1, "series_2": series_2})
        series_df = series_df.dropna()
        series_1 = series_df["series_1"]
        series_2 = series_df["series_2"]
        
        # ======= III. Return Processed Data =======    
        processed_data = (series_1, series_2)

        return processed_data

    #?____________________________________________________________________________________ #
    def get_feature(
        self, 
        data: Union[tuple, pd.DataFrame],
        window: int,
        residuals_weights: list,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.DataFrame:
        """
        Computes Ornstein-Uhlenbeck parameters for the spread over a rolling window.

        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
            - residuals_weights (list): Weights for spread calculation.
                If None: Use cointegration residuals. If [w, b]: Use spread = series_1 - w * series_2 - b
            - window (int): Rolling window size for cointegration testing.
            - smoothing_method (str): See process_data.
            - window_smooth (int): See process_data.
            - lambda_smooth (float): See process_data.

        Returns:
            - features_df (pd.DataFrame): Contains OU mu, theta, sigma, and half-life.
        """
        # ======= I. Process Data =======
        processed_data = self.process_data(data=data)
        series_1 = processed_data[0]
        series_2 = processed_data[1]

        # ======= II. Apply Smoothing if Needed =======
        if smoothing_method is not None:
            series_1 = self.smooth_data(data=series_1, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_2 = self.smooth_data(data=series_2, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)

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
                _, _, _, _, residuals = utils.get_cointegration(series_1=series1_window, series_2=series2_window)
            else: 
                residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
            
            # IV.3 Perform Ornstein-Uhlenbeck Estimation
            mu, theta, sigma, half_life = utils.get_OU_estimation(series=residuals)
            
            # IV.4 Store Results
            mu_values[i] = mu
            theta_values[i] = theta
            sigma_values[i] = sigma
            half_life_values[i] = half_life
        
        # ======= V. Create the Final DataFrame ========
        if residuals_weights is None:
            residuals_weights = [None, None]

        index = series_1.index[window:]
        features_df = pd.DataFrame({
            f"{self.name}_mu_{window}_b{residuals_weights[0]}a{residuals_weights[1]}_{smoothing_method}_{window_smooth}_{lambda_smooth}": mu_values,
            f"{self.name}_theta_{window}_b{residuals_weights[0]}a{residuals_weights[1]}_{smoothing_method}_{window_smooth}_{lambda_smooth}": theta_values,
            f"{self.name}_sigma_{window}_b{residuals_weights[0]}a{residuals_weights[1]}_{smoothing_method}_{window_smooth}_{lambda_smooth}": sigma_values,
            f"{self.name}_half_life_{window}_b{residuals_weights[0]}a{residuals_weights[1]}_{smoothing_method}_{window_smooth}_{lambda_smooth}": half_life_values,
        }, index=index)
        
        return features_df

#*____________________________________________________________________________________ #
class KalmanOU_feature(com.Feature):
    """
    Rolling Kalman Filter-Based Ornstein-Uhlenbeck Feature Extraction

    This class estimates the hidden states of an Ornstein-Uhlenbeck (OU) process using a Kalman filter over a rolling window. 
    It includes the filtered state and variance estimates.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving kalman Ornstein-Uhlenbeck estimation features over a rolling window
    """
    def __init__(
        self, 
        name: str = "kalmanOU", 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the kalmanOU_feature object with a pair of input series.

        Parameters:
            - name (str): Feature name.
            - n_jobs (int): Number of parallel jobs to use.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
        )

    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        window: list = [5, 10, 30, 60], 
        residuals_weights: list = [None, [1, 0]], 
        smooth_coefficient: list = [0.2, 0.5, 0.8], 
        smoothing_method: list = [None, "ewma", "average"], 
        window_smooth: list = [5, 10], 
        lambda_smooth: list = [0.1, 0.2, 0.5]
    ) -> Self:
        """
        Sets the parameter grid for Kalman OU feature extraction.

        Parameters:
            - window (list): Rolling window sizes for parameter estimation.
            - residuals_weights (list): List of residual computation modes. 
                                        If None, use cointegration residuals; 
                                        otherwise, use fixed [weight, intercept].
            - smooth_coefficient (list): Kalman filter update coefficient (0-1), 
                                         controls how fast the filter adapts.
            - smoothing_method (list): Smoothing method to apply before estimation.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA smoothing.
        """
        self.params = {
            "window": window,
            "residuals_weights": residuals_weights,
            "smooth_coefficient": smooth_coefficient,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }
        
        return self

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: Union[tuple, pd.DataFrame],
    ) -> tuple:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
        
        Returns:
            - processed_data (tuple(pd.Series)): The two series to be used for cointegration testing.
        """
        # ======= I. Extract Series =======
        if isinstance(data, pd.DataFrame):
            nb_series = data.shape[1]
            if nb_series != 2:
                raise ValueError(f"DataFrame must have exactly 2 columns, but got {nb_series}.")
            
            series_1 = data.iloc[:, 0]
            series_2 = data.iloc[:, 1]
        
        elif isinstance(data, tuple) and len(data) == 2:
            series_1 = data[0]
            series_2 = data[1]
        else:
            raise ValueError("Data must be either a tuple of two series or a DataFrame with two columns.")
        
        # ======= II. Ensure Series have the same indexation =======
        series_df = pd.DataFrame({"series_1": series_1, "series_2": series_2})
        series_df = series_df.dropna()
        series_1 = series_df["series_1"]
        series_2 = series_df["series_2"]
        
        # ======= III. Return Processed Data =======    
        processed_data = (series_1, series_2)

        return processed_data

    #?____________________________________________________________________________________ #
    def get_feature(
        self, 
        data: Union[tuple, pd.DataFrame],
        window: int,
        residuals_weights: list,
        smooth_coefficient: float,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.DataFrame:
        """
        Computes filtered state and variance estimates of an OU process using the Kalman filter over a rolling window.

        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
            - window (int): Rolling window size for OU Kalman estimation.
            - residuals_weights (list): Weights for spread calculation.
                If None: Use cointegration residuals.
                If [w, b]: Use spread = series_1 - w * series_2 - b
            - smooth_coefficient (float): Smoothing coefficient (0-1) for Kalman update.
            - smoothing_method (str): See process_data.
            - window_smooth (int): See process_data.
            - lambda_smooth (float): See process_data.

        Returns:
            - features_df (pd.DataFrame): Contains Kalman-filtered OU states and variances.
        """
        # ======= I. Process Data =======
        processed_data = self.process_data(data=data)
        series_1 = processed_data[0]
        series_2 = processed_data[1]

        # ======= II. Apply Smoothing if Needed =======
        if smoothing_method is not None:
            series_1 = self.smooth_data(data=series_1, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_2 = self.smooth_data(data=series_2, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)

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
                _, _, _, _, residuals = utils.get_cointegration(series_1=series1_window, series_2=series2_window)
            else: 
                residuals = series1_window - residuals_weights[0] * series2_window - residuals_weights[1]
            
            # IV.3 Perform Ornstein-Uhlenbeck Estimation
            filtered_states, variances = utils.kalmanOU_smoothing(series=residuals, smooth_coefficient=smooth_coefficient)
            
            # IV.4 Store Results
            state_values[i] = filtered_states.iloc[-1]
            variance_values[i] = variances.iloc[-1]
        
        # ======== V. Create the Final DataFrame ========
        if residuals_weights is None:
            residuals_weights = [None, None]

        index = series_1.index[window:]
        features_df = pd.DataFrame({
            f"{self.name}_state_{window}_b{residuals_weights[0]}a{residuals_weights[1]}_{smoothing_method}_{window_smooth}_{lambda_smooth}": state_values,
            f"{self.name}_variance_{window}_b{residuals_weights[0]}a{residuals_weights[1]}_{smoothing_method}_{window_smooth}_{lambda_smooth}": variance_values,
        }, index=index)
        
        return features_df

