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
    """
    Rolling Cointegration Feature Extraction

    This class computes cointegration statistics between two time series using 
    the Engle-Granger two-step method. Over a rolling window, it estimates 
    beta, intercept, ADF/KPSS p-values, and the last residual value.

    Inherits from the Feature base class.
    """
    def __init__(
        self, 
        data: Union[tuple, pd.DataFrame], 
        name: str = "cointegration", 
        n_jobs: int = 1
    ):
        """
        Initializes the Cointegration_feature object with a pair of input series.

        Parameters:
            - data (tuple or pd.DataFrame): A tuple of two series or a DataFrame with two columns.
            - name (str): Feature name.
            - n_jobs (int): Number of parallel jobs to use.
        """
        super().__init__(
            data=data, 
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
    ):
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ) -> pd.DataFrame:
        """
        Applies optional smoothing to the input time series.

        Parameters:
            - smoothing_method (str): "ewma", "average", or None.
            - window_smooth (int): Window size for smoothing.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - processed_data (pd.DataFrame): A DataFrame containing smoothed series_1 and series_2.
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
        Computes rolling cointegration statistics between two series.

        Parameters:
            - window (int): Rolling window size for cointegration testing.
            - smoothing_method (str): See process_data.
            - window_smooth (int): See process_data.
            - lambda_smooth (float): See process_data.

        Returns:
            - features_df (pd.DataFrame): Contains beta, intercept, ADF p-value, KPSS p-value, and residuals.
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
    """
    Rolling Ornstein-Uhlenbeck Process Feature Extraction

    This class estimates Ornstein-Uhlenbeck (OU) parameters over a rolling window.
    The OU process is often used to model mean-reverting behavior, particularly in 
    spread trading strategies. Features include the long-term mean (mu), speed of 
    reversion (theta), volatility (sigma), and the half-life of mean reversion.

    The spread can be computed using either fixed weights or dynamic residuals 
    from cointegration tests.

    Inherits from the Feature base class.
    """
    def __init__(
        self, 
        data: Union[tuple, pd.DataFrame], 
        name: str = "OrnsteinUhlenbeck", 
        n_jobs: int = 1
    ):
        """
        Initializes the OU_feature object with a pair of input series.

        Parameters:
            - data (tuple or pd.DataFrame): A tuple of two series or a DataFrame with two columns.
            - name (str): Feature name.
            - n_jobs (int): Number of parallel jobs to use.
        """
        super().__init__(
            data=data, 
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
    ):
        """
        Sets the parameter grid for OU feature extraction.

        Parameters:
            - window (list): Rolling window sizes for OU parameter estimation.
            - residuals_weights (list): List of residual computation modes. 
                                        If None, use cointegration residuals; 
                                        otherwise, use fixed [weight, intercept].
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ) -> pd.DataFrame:
        """
        Applies optional smoothing to the input time series.

        Parameters:
            - smoothing_method (str): "ewma", "average", or None.
            - window_smooth (int): Window size for smoothing.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - processed_data (pd.DataFrame): A DataFrame containing smoothed series_1 and series_2.
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
        Computes Ornstein-Uhlenbeck parameters for the spread over a rolling window.

        Parameters:
            - window (int): Rolling window size for OU parameter estimation.
            - residuals_weights (list): Weights for spread calculation.
                If None: Use cointegration residuals.
                If [w, b]: Use spread = series_1 - w * series_2 - b
            - smoothing_method (str): See process_data.
            - window_smooth (int): See process_data.
            - lambda_smooth (float): See process_data.

        Returns:
            - features_df (pd.DataFrame): Contains OU mu, theta, sigma, and half-life.
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
    """
    Rolling Kalman Filter-Based Ornstein-Uhlenbeck Feature Extraction

    This class estimates the hidden states and variances of an Ornstein-Uhlenbeck (OU) 
    process using a Kalman filter over a rolling window. It is designed to capture the 
    evolving dynamics of mean-reverting relationships, especially in pairs trading strategies.

    The spread can be computed either from cointegration residuals or using fixed linear 
    weights between the two series. Smoothing can be optionally applied to the input series 
    before feature extraction.

    Inherits from the Feature base class.
    """
    def __init__(
        self, 
        data: Union[tuple, pd.DataFrame], 
        name: str = "kalmanOU", 
        n_jobs: int = 1
    ):
        """
        Initializes the kalmanOU_feature object with a pair of input series.

        Parameters:
            - data (tuple or pd.DataFrame): A tuple of two series or a DataFrame with two columns.
            - name (str): Feature name.
            - n_jobs (int): Number of parallel jobs to use.
        """
        super().__init__(
            data=data, 
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
    ):
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ) -> pd.DataFrame:
        """
        Applies optional smoothing to the input time series.

        Parameters:
            - smoothing_method (str): "ewma", "average", or None.
            - window_smooth (int): Window size for smoothing.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - processed_data (pd.DataFrame): A DataFrame containing smoothed series_1 and series_2.
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
        Computes filtered state and variance estimates of an OU process 
        using the Kalman filter over a rolling window.

        Parameters:
            - window (int): Rolling window size for OU Kalman estimation.
            - residuals_weights (list): Weights for spread calculation.
                If None: Use cointegration residuals.
                If [w, b]: Use spread = series_1 - w * series_2 - b
            - smooth_coefficient (float): Smoothing coefficient (0–1) for Kalman update.
            - smoothing_method (str): See process_data.
            - window_smooth (int): See process_data.
            - lambda_smooth (float): See process_data.

        Returns:
            - features_df (pd.DataFrame): Contains Kalman-filtered OU states and variances.
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

