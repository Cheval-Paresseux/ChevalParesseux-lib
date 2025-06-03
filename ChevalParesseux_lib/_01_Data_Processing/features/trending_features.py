from ..features import common as com
from ... import utils

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Self, Union



#! ==================================================================================== #
#! ============================= Series Trending Features ============================= #
class Momentum_feature(com.Feature):
    """
    Rolling Momentum Feature

    This class computes the rolling momentum of a time series, with optional smoothing applied to the series before the computation.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving momentum feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "momentum" , 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the momentum_feature object with the input series.

        Parameters:
            - data (pd.Series): The raw input time series.
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
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ) -> Self:
        """
        Sets the parameter grid for momentum feature extraction.

        Parameters:
            - window (list): Rolling window sizes for momentum calculation.
            - smoothing_method (list): Type of smoothing to apply before calculation.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA.
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
        data: pd.Series,
    ) -> pd.Series:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The resetted index series.
        """
        processed_data = data.copy()
        processed_data.reset_index(drop=True, inplace=True)

        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        data: pd.Series,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.Series:
        """
        Computes the rolling momentum from the smoothed series.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - window (int): Rolling window size.
            - smoothing_method (str): Smoothing method to apply.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - rolling_momentum (pd.Series): Series of rolling momentum values.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the moving momentum =======
        rolling_momentum = processed_series.rolling(window=window ).apply(utils.get_momentum, raw=False)
        
        # ======= III. Convert to pd.Series and Center =======
        rolling_momentum = pd.Series(rolling_momentum, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_momentum.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_momentum.index = data.index

        return rolling_momentum

#*____________________________________________________________________________________ #
class Z_momentum_feature(com.Feature):
    """
    Rolling Z-Momentum Feature

    This class computes the rolling Z-momentum of a time series. Z-momentum is a normalized momentum value (e.g., z-score of returns or momentum),
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving Z-momentum feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "Z_momentum" , 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the Z_momentum_feature object with the input series.

        Parameters:
            - data (pd.Series): The raw input time series.
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
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ) -> Self:
        """
        Sets the parameter grid for Z-momentum feature extraction.

        Parameters:
            - window (list): Rolling window sizes for Z-momentum calculation.
            - smoothing_method (list): Type of smoothing to apply before calculation.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA.
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
        data: pd.Series,
    ) -> pd.Series:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The resetted index series.
        """
        processed_data = data.copy()
        processed_data.reset_index(drop=True, inplace=True)

        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        data: pd.Series,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the rolling Z-momentum from the smoothed series.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - window (int): Rolling window size.
            - smoothing_method (str): Smoothing method to apply.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - rolling_Z_momentum (pd.Series): Series of rolling Z-momentum values.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the moving Z-momentum =======
        rolling_Z_momentum = processed_series.rolling(window=window).apply(utils.get_Z_momentum, raw=False)
        
        # ======= III. Convert to pd.Series and Center =======
        rolling_Z_momentum = pd.Series(rolling_Z_momentum, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_Z_momentum.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_Z_momentum.index = data.index

        return rolling_Z_momentum

#*____________________________________________________________________________________ #
class Linear_tempReg_feature(com.Feature):
    """
    Rolling Linear Temporal Regression Feature

    This class computes a rolling linear regression statistics over a time series. 
    For each window, it fits a linear model and extracts the slope, t-statistic, p-value, and R-squared.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving linear temporal regression feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "linear_tempreg" , 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the linear_tempReg_feature object with the input time series.

        Parameters:
            - data (pd.Series): The raw input time series.
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
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ) -> Self:
        """
        Sets the parameter grid for the rolling linear regression feature extraction.

        Parameters:
            - window (list): Rolling window sizes for regression.
            - smoothing_method (list): Type of smoothing to apply before regression.
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
        data: pd.Series,
    ) -> pd.Series:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        ________
        N.B: The feature does not require preprocessing, but this method is kept for consistency.
        """
        return data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        data: pd.Series,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.DataFrame:
        """
        Computes rolling linear regression statistics (slope, t-stat, p-value, R-squared)
        on the smoothed series over the specified window.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - window (int): Rolling window size for regression.
            - smoothing_method (str): Smoothing method to apply before regression.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA smoothing.

        Returns:
            - features_df (pd.DataFrame): DataFrame containing regression statistics.
        """
        # ======= 0. Intermediate functions =======
        def compute_regression(
            series: pd.Series, 
            start_idx: int, 
            window: int
        ) -> tuple:
            
            current_window = series.iloc[start_idx - window + 1: start_idx + 1]
            intercept, coefficients, metrics = utils.get_simple_TempReg(series=current_window)
            
            return start_idx, intercept, coefficients, metrics
            

        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the rolling regression statistics =======
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_regression)(processed_series, i, window)
            for i in range(window - 1, len(processed_series))
        )

        # ======= III. Convert to pd.Series =======
        rolling_slope = pd.Series({i: coeffs[0] for i, _, coeffs, _ in results}) 
        rolling_r2 = pd.Series({i: metrics['r2'] for i, _, _, metrics in results})
        
        # ======= IV. Rearrange the index =======
        rolling_slope.index = processed_series.index[window - 1:]
        rolling_r2.index = processed_series.index[window - 1:]
        
        # ======= V. Center =======
        rolling_slope = rolling_slope / (processed_series.loc[rolling_slope.index] + 1e-8)
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"{self.name}_slope_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_slope,
            f"{self.name}_r2_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_r2,
        })
        
        return features_df

#*____________________________________________________________________________________ #
class Nonlinear_tempReg_feature(com.Feature):
    """
    Rolling Nonlinear (Quadratic) Temporal Regression Feature

    This class computes nonlinear regression statistics over a time series.
    For each window, it fits a quadratic model and extracts the slope, acceleration, t-statistic, p-value, and R-squared.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving non-linear temporal regression feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "nonlinear_tempreg" , 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the nonlinear_tempReg_feature object with the input series.

        Parameters:
            - data (pd.Series): The raw input time series.
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
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ) -> Self:
        """
        Sets the parameter grid for nonlinear regression feature extraction.

        Parameters:
            - window (list): Rolling window sizes for nonlinear regression.
            - smoothing_method (list): Type of smoothing to apply before calculation.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA.
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
        data: pd.Series,
    ) -> pd.Series:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        ________
        N.B: The feature does not require preprocessing, but this method is kept for consistency.
        """
        return data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        data: pd.Series,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.DataFrame:
        """
        Computes rolling nonlinear regression features from the smoothed series.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - window (int): Rolling window size.
            - smoothing_method (str): Smoothing method to apply.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - features_df (pd.DataFrame): DataFrame with slope, acceleration, 
              t-statistic, p-value, and R-squared for each window.
        """
        # ======= 0. Intermediate functions =======
        def compute_regression(
            series: pd.Series, 
            start_idx: int, 
            window: int
        ) -> tuple:
            
            current_window = series.iloc[start_idx - window + 1: start_idx + 1]
            intercept, coefficients, metrics = utils.get_quad_TempReg(series=current_window)
            
            return start_idx, intercept, coefficients, metrics
            

        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the rolling regression statistics =======
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_regression)(processed_series, i, window)
            for i in range(window - 1, len(processed_series))
        )

        # ======= III. Convert to pd.Series =======
        rolling_slope = pd.Series({i: coeffs[0] for i, _, coeffs, _ in results}) 
        rolling_acceleration = pd.Series({i: coeffs[1] for i, _, coeffs, _ in results})
        rolling_r2 = pd.Series({i: metrics['r2'] for i, _, _, metrics in results})
        
        # ======= IV. Rearrange the index =======
        rolling_slope.index = processed_series.index[window - 1:]
        rolling_acceleration.index = processed_series.index[window - 1:]
        rolling_r2.index = processed_series.index[window - 1:]
        
        # ======= V. Center =======
        rolling_slope = rolling_slope / (processed_series.loc[rolling_slope.index] + 1e-8)
        rolling_acceleration = rolling_acceleration / (processed_series.loc[rolling_slope.index] + 1e-8)
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"{self.name}_slope_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_slope,
            f"{self.name}_acceleration_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_acceleration,
            f"{self.name}_r2_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_r2,
        })

        return features_df

#*____________________________________________________________________________________ #
class Hurst_exponent_feature(com.Feature):
    """
    Rolling Hurst Exponent Feature

    This class computes the Hurst exponent of a time series, which is a measure of the long-term memory of the time series.
    Values > 0.5 suggest trend-following behavior and values < 0.5 suggest mean-reversion.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving hurst exponent feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "hurst" , 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the hurst_exponent_feature object with the input series.

        Parameters:
            - data (pd.Series): The raw input time series.
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
        power: list = [3, 4, 5, 6],
        smoothing_method: list = [None, "ewma", "average"],
        window_smooth: list = [5, 10],
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ) -> Self:
        """
        Sets the parameter grid for Hurst exponent feature extraction.

        Parameters:
            - power (list): Exponents of 2 used for window sizing (e.g., 2^3 = 8).
            - smoothing_method (list): Type of smoothing to apply before calculation.
            - window_smooth (list): Window sizes for smoothing filters.
            - lambda_smooth (list): Decay factors for EWMA smoothing.
        """
        self.params = {
            "power": power,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }

        return self

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: pd.Series,
    ) -> pd.Series:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        ________
        N.B: The feature does not require preprocessing, but this method is kept for consistency.
        """
        return data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        data: pd.Series,
        power: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.DataFrame:
        """
        Computes rolling Hurst exponent values, along with t-statistics and p-values.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - power (int): Power of 2 used to define the rolling window size.
            - smoothing_method (str): Smoothing method applied before computation.
            - window_smooth (int): Window size for smoothing.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - features_df (pd.DataFrame): DataFrame containing:
                - Hurst exponent (centered around 0.5)
                - t-statistic of regression slope
                - p-value of regression slope
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the returns =======
        prices_array = np.array(processed_series)
        returns_array = prices_array[1:] / prices_array[:-1] - 1

        # ======= II. Compute the Hurst Exponent =======
        n = 2**power

        hursts = np.array([])
        tstats = np.array([])
        pvalues = np.array([])
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

            model = utils.OLS_regression()
            model.fit(X, Y)
            
            hurst = model.coefficients[0]
            tstat = model.metrics['significance']['t_stat'].iloc[0]
            pvalue = model.metrics['significance']['p_value'].iloc[0]
            
            hursts = np.append(hursts, hurst)
            tstats = np.append(tstats, tstat)
            pvalues = np.append(pvalues, pvalue)

        # ======= III. Convert to pd.Series and Center =======
        hursts = pd.Series([np.nan] * n + list(hursts), index=processed_series.index) - 0.5
        tstats = pd.Series([np.nan] * n + list(tstats), index=processed_series.index)
        pvalues = pd.Series([np.nan] * n + list(pvalues), index=processed_series.index)

        tstats_mean = tstats.rolling(window=252).mean()
        tstats = tstats - tstats_mean

        pvalues_mean = pvalues.rolling(window=252).mean()
        pvalues = pvalues - pvalues_mean
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"{self.name}_exponent{power}_{smoothing_method}_{window_smooth}_{lambda_smooth}": hursts,
            f"{self.name}_tstat_{power}_{smoothing_method}_{window_smooth}_{lambda_smooth}": tstats,
            f"{self.name}_pvalue_{power}_{smoothing_method}_{window_smooth}_{lambda_smooth}": pvalues,
        })
        
        return features_df

#*____________________________________________________________________________________ #
class Kama_feature(com.Feature):
    """
    Kaufman Adaptive Moving Average Feature

    This class computes the Kaufman Adaptive Moving Average (KAMA) of a time series, with optional pre-smoothing filters.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving average feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "kama" , 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the Kama_feature object with input data, name, and parallel jobs.
        
        Parameters:
            - name (str): Name of the feature, used in column labeling.
            - n_jobs (int): Number of jobs to run in parallel for feature extraction.
        """
        super().__init__(name=name, n_jobs=n_jobs)
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        window: list = [5, 10, 30, 60],
        smoothing_method: list = [None, "ewma", "average"],
        window_smooth: list = [5, 10],
        lambda_smooth: list = [0.1, 0.2, 0.5],
        fastest_window: list = [2, 5, 10],
        slowest_window: list = [20, 30],
    ) -> Self:
        """
        Sets the parameter grid for the KAMA feature extraction.

        Parameters:
            - window (list): Rolling window sizes for momentum calculation.
            - smoothing_method (list): Type of smoothing to apply before calculation.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA.
            - fastest_window (list): Fastest window sizes for KAMA calculation.
            - slowest_window (list): Slowest window sizes for KAMA calculation.
        """
        self.params = {
            "window": window,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
            "fastest_window": fastest_window,
            "slowest_window": slowest_window,
        }

        return self

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: pd.Series,
    ) -> pd.Series:
        """
        Preprocesses the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The resetted index series.
        """
        processed_data = data.copy()
        processed_data.reset_index(drop=True, inplace=True)

        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        data: pd.Series,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
        fastest_window: int,
        slowest_window: int,
    ) -> pd.Series:
        """
        Computes the normalized Kaufman Adaptive Moving Average (KAMA) of the processed series.
        
        Parameters:
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the KAMA calculation.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.
            - fastest_window (int): Fastest window size for KAMA calculation.
            - slowest_window (int): Slowest window size for KAMA calculation.
        
        Returns:
            - rolling_kama (pd.Series): The resulting normalized KAMA feature.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the rolling kama =======
        rolling_kama = processed_series.rolling(window=window ).apply(utils.get_kama, args=(fastest_window, slowest_window), raw=False)
        
        # ======= III. Convert to pd.Series and Center =======
        rolling_kama = (pd.Series(rolling_kama, index=processed_series.index) / (processed_series + 1e-8)) - 1
        
        # ======= IV. Change Name =======
        rolling_kama.name = f"{self.name}_f{fastest_window}_s{slowest_window}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_kama.index = data.index

        return rolling_kama

#*____________________________________________________________________________________ #
class StochasticRSI_feature(com.Feature):
    """
    Stochastic RSI Feature
    
    This class computes the Stochastic RSI of a time series, which is a momentum oscillator that measures the level of RSI relative to its high-low range over a specified period.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving Stochastic RSI feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "stochastic_rsi", 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the StochasticRSI_feature object with the input series.
        
        Parameters:
            - name (str): Name of the feature, used in column labeling.
            - n_jobs (int): Number of jobs to run in parallel for feature extraction.
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
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ) -> Self:
        """
        Sets the parameter grid for momentum feature extraction.

        Parameters:
            - window (list): Rolling window sizes for momentum calculation.
            - smoothing_method (list): Type of smoothing to apply before calculation.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA.
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
        data: pd.Series,
    ) -> pd.Series:
        """
        Preprocesses the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The resetted index series.
        """
        processed_data = data.copy()
        processed_data.reset_index(drop=True, inplace=True)

        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        data: pd.Series,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.Series:
        """
        Computes the Stochastic RSI of the processed series.
        
        Parameters:
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the Stochastic RSI calculation.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.
        
        Returns:
            - rolling_stoch_rsi (pd.Series): The resulting Stochastic RSI feature.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the rolling stochastic RSI =======
        rolling_stoch_rsi = processed_series.rolling(window=window ).apply(utils.get_stochastic_rsi, raw=False)
        
        # ======= III. Convert to pd.Series =======
        rolling_stoch_rsi = pd.Series(rolling_stoch_rsi, index=processed_series.index) 
        
        # ======= IV. Change Name =======
        rolling_stoch_rsi.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_stoch_rsi.index = data.index

        return rolling_stoch_rsi

#*____________________________________________________________________________________ #
class RSI_feature(com.Feature):
    """
    Relative Strength Index (RSI) Feature
    
    This class computes the Relative Strength Index (RSI) of a time series, which is a momentum oscillator that measures the speed and change of price movements.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving RSI feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "rsi", 
        n_jobs: int = 1
    ) -> None:
        """
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
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ) -> Self:
        """
        Sets the parameter grid for momentum feature extraction.

        Parameters:
            - window (list): Rolling window sizes for momentum calculation.
            - smoothing_method (list): Type of smoothing to apply before calculation.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA.
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
        data: pd.Series,
    ) -> pd.Series:
        """
        Preprocesses the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The resetted index series.
        """
        processed_data = data.copy()
        processed_data.reset_index(drop=True, inplace=True)

        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        data: pd.Series,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.Series:
        """
        Computes the Relative Strength Index (RSI) of the processed series.
        
        Parameters:
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the RSI calculation.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.
        
        Returns:
            - rolling_rsi (pd.Series): The resulting RSI feature, centered around 0.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the rolling RSI =======
        rolling_rsi = processed_series.rolling(window=window ).apply(utils.get_relative_strength_index, raw=False)
        
        # ======= III. Convert to pd.Series =======
        rolling_rsi = pd.Series(rolling_rsi, index=processed_series.index) 
        
        # ======= IV. Change Name =======
        rolling_rsi.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_rsi.index = data.index

        return rolling_rsi

#*____________________________________________________________________________________ #
class EhlersFisher_feature(com.Feature):
    """
    Ehlers Fisher Transform Feature
    
    This class computes the Ehlers Fisher Transform of two time series, which is a method to convert prices into a Gaussian distribution.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : preprocess the input data.
        - get_feature : compute the Ehlers Fisher Transform feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "ehlers_fisher", 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the EhlersFisher_feature object with the input series.

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
        Processes the input data to extract two time series for the Ehlers Fisher Transform.
        
        Parameters:
            - data (Union[tuple, pd.DataFrame]): Input data containing two time series.
        
        Returns:
            - processed_data (tuple): A tuple containing the two processed time series.
        """
        # ======= I. Extract Series =======
        if isinstance(data, pd.DataFrame):
            nb_series = data.shape[1]
            if nb_series != 2:
                raise ValueError(f"DataFrame must have exactly 2 columns, but got {nb_series}.")
            
            series_high = data.iloc[:, 0]
            series_low = data.iloc[:, 1]
        
        elif isinstance(data, tuple) and len(data) == 2:
            series_high = data[0]
            series_low = data[1]
        else:
            raise ValueError("Data must be either a tuple of two series or a DataFrame with two columns.")
        
        # ======= II. Ensure Series have the same indexation =======
        series_df = pd.DataFrame({"series_high": series_high, "series_low": series_low})
        series_df = series_df.dropna()
        series_high = series_df["series_high"]
        series_low = series_df["series_low"]
        
        # ======= III. Return Processed Data =======    
        processed_data = (series_high, series_low)

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
        Computes the Ehlers Fisher Transform feature from two time series.
        
        Parameters:
            - data (Union[tuple, pd.DataFrame]): Input data containing two time series.
            - window (int): Rolling window size for the Fisher transform.
            - smoothing_method (str): Smoothing method to apply before testing.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA smoothing.
        
        Returns:
            - features_df (pd.DataFrame): DataFrame containing the Ehlers Fisher Transform values.
        """
        # ======= I. Process Data =======
        processed_data = self.process_data(data=data)
        series_high = processed_data[0]
        series_low = processed_data[1]

        # ======= II. Apply Smoothing if Needed =======
        if smoothing_method is not None:
            series_high = self.smooth_data(data=series_high, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_low = self.smooth_data(data=series_low, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)

        # ======= II. Ensure the window is not too large =======
        num_obs = len(series_high) - window
        if num_obs <= 0:
            raise ValueError(f"Window size {window} is too large for the given data length {len(series_high)}.")
        
        # ======= III. Initialize Output Arrays =======
        elhers_fisher_values = np.full(num_obs, np.nan)

        # ======== IV. Iterate Over Observations ========
        for i in range(num_obs):
            # IV.1 Extract Time Windows
            series_high_window = series_high.iloc[i : i + window]
            series_low_window = series_low.iloc[i : i + window]

            # IV.2 Perform Elhers Fisher transform Test
            elhers_fisher = utils.get_ehlers_fisher_transform(
                series_high=series_high_window, 
                series_low=series_low_window
            )

            # IV.3 Store Results
            elhers_fisher_values[i] = elhers_fisher

        # ======== V. Create the Final DataFrame ========
        index = series_high.index[window:]
        features_df = pd.DataFrame({
            f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": elhers_fisher_values,
        }, index=index)

        return features_df

#*____________________________________________________________________________________ #
class Oscillator_feature(com.Feature):
    """
    Oscillator Feature
    
    This class computes a rolling oscillator feature from a time series, which can be used to identify overbought or oversold conditions.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : preprocess the input data.
        - get_feature : compute the rolling oscillator feature over a specified window
    """
    def __init__(
        self, 
        name: str = "oscillator", 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the Oscillator_feature object with the input series.
        
        Parameters:
            - name (str): Name of the feature, used in column labeling.
            - n_jobs (int): Number of jobs to run in parallel for feature extraction.
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
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ) -> Self:
        """
        Sets the parameter grid for momentum feature extraction.

        Parameters:
            - window (list): Rolling window sizes for momentum calculation.
            - smoothing_method (list): Type of smoothing to apply before calculation.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA.
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
        data: pd.Series,
    ) -> pd.Series:
        """
        Preprocesses the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The resetted index series.
        """
        processed_data = data.copy()
        processed_data.reset_index(drop=True, inplace=True)

        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        data: pd.Series,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.Series:
        """
        Computes the rolling oscillator feature from the processed series.
        
        Parameters:
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the oscillator calculation.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.
        
        Returns:
            - rolling_oscillator (pd.Series): The resulting oscillator feature, centered around 0.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the rolling oscillator =======
        rolling_oscillator = processed_series.rolling(window=window ).apply(utils.get_oscillator, raw=False)
        
        # ======= III. Convert to pd.Series and Center =======
        rolling_oscillator = pd.Series(rolling_oscillator, index=processed_series.index) 
        
        # ======= IV. Change Name =======
        rolling_oscillator.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_oscillator.index = data.index

        return rolling_oscillator

#*____________________________________________________________________________________ #
class Vortex_feature(com.Feature):
    """
    Vortex Feature
    
    This class computes the Vortex Indicator, which is a trend-following indicator that measures the strength of a trend.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : preprocess the input data.
        - get_feature : compute the Vortex feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "vortex", 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the Vortex_feature object with the input series.
        
        Parameters:
            - name (str): Name of the feature, used in column labeling.
            - n_jobs (int): Number of jobs to run in parallel for feature extraction.
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
        Processes the input data to extract three time series for the Vortex Indicator.
        
        Parameters:
            - data (Union[tuple, pd.DataFrame]): Input data containing three time series.
        
        Returns:
            - processed_data (tuple): A tuple containing the three processed time series.
        """
        # ======= I. Extract Series =======
        if isinstance(data, pd.DataFrame):
            nb_series = data.shape[1]
            if nb_series != 3:
                raise ValueError(f"DataFrame must have exactly 3 columns, but got {nb_series}.")
            
            series_mid = data.iloc[:, 0]
            series_high = data.iloc[:, 1]
            series_low = data.iloc[:, 2]
        
        elif isinstance(data, tuple) and len(data) == 3:
            series_mid = data[0]
            series_high = data[1]
            series_low = data[2]
        else:
            raise ValueError("Data must be either a tuple of three series or a DataFrame with three columns.")
        
        # ======= II. Ensure Series have the same indexation =======
        series_df = pd.DataFrame({"series_mid": series_mid, "series_high": series_high, "series_low": series_low})
        series_df = series_df.dropna()
        series_mid = series_df["series_mid"]
        series_high = series_df["series_high"]
        series_low = series_df["series_low"]
        
        # ======= III. Return Processed Data =======    
        processed_data = (series_mid, series_high, series_low)

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
        Computes the Vortex Indicator feature from three time series.
        
        Parameters:
            - data (Union[tuple, pd.DataFrame]): Input data containing three time series.
            - window (int): Rolling window size for the Vortex calculation.
            - smoothing_method (str): Smoothing method to apply before testing.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA smoothing.
        
        Returns:
            - features_df (pd.DataFrame): DataFrame containing the Vortex Indicator values.
        """
        # ======= I. Process Data =======
        processed_data = self.process_data(data=data)
        series_mid = processed_data[0]
        series_high = processed_data[1]
        series_low = processed_data[2]

        # ======= II. Apply Smoothing if Needed =======
        if smoothing_method is not None:
            series_mid = self.smooth_data(data=series_mid, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_high = self.smooth_data(data=series_high, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_low = self.smooth_data(data=series_low, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)

        # ======= II. Ensure the window is not too large =======
        num_obs = len(series_high) - window
        if num_obs <= 0:
            raise ValueError(f"Window size {window} is too large for the given data length {len(series_high)}.")

        # ======= III. Initialize Output Arrays =======
        vortex_up_values = np.full(num_obs, np.nan)
        vortex_down_values = np.full(num_obs, np.nan)

        # ======== IV. Iterate Over Observations ========
        for i in range(num_obs):
            # IV.1 Extract Time Windows
            series_mid_window = series_mid.iloc[i : i + window]
            series_high_window = series_high.iloc[i : i + window]
            series_low_window = series_low.iloc[i : i + window]

            # IV.2 Perform Vortex Indicator Calculation
            vortex_up, vortex_down = utils.get_vortex(
                series_mid=series_mid_window,
                series_high=series_high_window, 
                series_low=series_low_window
            )

            # IV.3 Store Results
            vortex_up_values[i] = vortex_up
            vortex_down_values[i] = vortex_down

        # ======== V. Create the Final DataFrame ========
        index = series_high.index[window:]
        features_df = pd.DataFrame({
            f"{self.name}_up_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": vortex_up_values,
            f"{self.name}_down_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": vortex_down_values,
        }, index=index)

        return features_df

#*____________________________________________________________________________________ #
class Vigor_feature(com.Feature):
    """
    Vigor Feature
    
    This class computes the Vigor Index, which is a measure of the strength of a trend based on the relationship between open and close prices.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : preprocess the input data.
        - get_feature : compute the Vigor feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "vigor", 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the Vigor_feature object with the input series.
        
        Parameters:
            - name (str): Name of the feature, used in column labeling.
            - n_jobs (int): Number of jobs to run in parallel for feature extraction.
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
        Processes the input data to extract four time series for the Vigor Index.
        
        Parameters:
            - data (Union[tuple, pd.DataFrame]): Input data containing four time series (open, close, high, low).
        
        Returns:
            - processed_data (tuple): A tuple containing the four processed time series.
        """
        # ======= I. Extract Series =======
        if isinstance(data, pd.DataFrame):
            nb_series = data.shape[1]
            if nb_series != 4:
                raise ValueError(f"DataFrame must have exactly 4 columns, but got {nb_series}.")
            
            series_open = data.iloc[:, 0]
            series_close = data.iloc[:, 1]
            series_high = data.iloc[:, 2]
            series_low = data.iloc[:, 3]
        
        elif isinstance(data, tuple) and len(data) == 4:
            series_open = data[0]
            series_close = data[1]
            series_high = data[2]
            series_low = data[3]
        else:
            raise ValueError("Data must be either a tuple of four series or a DataFrame with four columns.")
        
        # ======= II. Ensure Series have the same indexation =======
        series_df = pd.DataFrame({"series_open": series_open, "series_close": series_close, "series_high": series_high, "series_low": series_low})
        series_df = series_df.dropna()
        
        series_open = series_df["series_open"]
        series_close = series_df["series_close"]
        series_high = series_df["series_high"]
        series_low = series_df["series_low"]
        
        # ======= III. Return Processed Data =======    
        processed_data = (series_open, series_close, series_high, series_low)

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
        Computes the Vigor Index feature from four time series (open, close, high, low).
        
        Parameters:
            - data (Union[tuple, pd.DataFrame]): Input data containing four time series.
            - window (int): Rolling window size for the Vigor Index calculation.
            - smoothing_method (str): Smoothing method to apply before testing.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA smoothing.
        
        Returns:
            - features_df (pd.DataFrame): DataFrame containing the Vigor Index values.
        """
        # ======= I. Process Data =======
        processed_data = self.process_data(data=data)
        series_open = processed_data[0]
        series_close = processed_data[1]
        series_high = processed_data[2]
        series_low = processed_data[3]

        # ======= II. Apply Smoothing if Needed =======
        if smoothing_method is not None:
            series_open = self.smooth_data(data=series_open, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_close = self.smooth_data(data=series_close, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_high = self.smooth_data(data=series_high, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_low = self.smooth_data(data=series_low, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)

        # ======= II. Ensure the window is not too large =======
        num_obs = len(series_high) - window
        if num_obs <= 0:
            raise ValueError(f"Window size {window} is too large for the given data length {len(series_high)}.")

        # ======= III. Initialize Output Arrays =======
        vigor_values = np.full(num_obs, np.nan)

        # ======== IV. Iterate Over Observations ========
        for i in range(num_obs):
            # IV.1 Extract Time Windows
            series_open_window = series_open.iloc[i : i + window]
            series_close_window = series_close.iloc[i : i + window]
            series_high_window = series_high.iloc[i : i + window]
            series_low_window = series_low.iloc[i : i + window]

            # IV.2 Perform Vigor Index Calculation
            vigor_index = utils.get_vigor(
                series_open=series_open_window,
                series_close=series_close_window,
                series_high=series_high_window, 
                series_low=series_low_window
            )

            # IV.3 Store Results
            vigor_values[i] = vigor_index

        # ======== V. Create the Final DataFrame ========
        index = series_high.index[window:]
        features_df = pd.DataFrame({
            f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": vigor_values,
        }, index=index)

        return features_df

#*____________________________________________________________________________________ #
class StochasticOscillator_feature(com.Feature):
    """
    Stochastic Oscillator Feature
    
    This class computes the Stochastic Oscillator, which is a momentum indicator that compares a particular closing price of a security to a range of its prices over a certain period.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : preprocess the input data.
        - get_feature : compute the Stochastic Oscillator feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "stochoscillator", 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the StochasticOscillator_feature object with the input series.
        
        Parameters:
            - name (str): Name of the feature, used in column labeling.
            - n_jobs (int): Number of jobs to run in parallel for feature extraction.
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
        Processes the input data to extract three time series for the Stochastic Oscillator.
        
        Parameters:
            - data (Union[tuple, pd.DataFrame]): Input data containing three time series (mid, high, low).
        
        Returns:
            - processed_data (tuple): A tuple containing the three processed time series.
        """
        # ======= I. Extract Series =======
        if isinstance(data, pd.DataFrame):
            nb_series = data.shape[1]
            if nb_series != 3:
                raise ValueError(f"DataFrame must have exactly 3 columns, but got {nb_series}.")
            
            series_mid = data.iloc[:, 0]
            series_high = data.iloc[:, 1]
            series_low = data.iloc[:, 2]
        
        elif isinstance(data, tuple) and len(data) == 3:
            series_mid = data[0]
            series_high = data[1]
            series_low = data[2]
        else:
            raise ValueError("Data must be either a tuple of three series or a DataFrame with three columns.")
        
        # ======= II. Ensure Series have the same indexation =======
        series_df = pd.DataFrame({"series_mid": series_mid, "series_high": series_high, "series_low": series_low})
        series_df = series_df.dropna()
        series_mid = series_df["series_mid"]
        series_high = series_df["series_high"]
        series_low = series_df["series_low"]
        
        # ======= III. Return Processed Data =======    
        processed_data = (series_mid, series_high, series_low)

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
        Computes the Stochastic Oscillator feature from three time series (mid, high, low).
        
        Parameters:
            - data (Union[tuple, pd.DataFrame]): Input data containing three time series.
            - window (int): Rolling window size for the Stochastic Oscillator calculation.
            - smoothing_method (str): Smoothing method to apply before testing.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA smoothing.
        
        Returns:
            - features_df (pd.DataFrame): DataFrame containing the Stochastic Oscillator values.
        """
        # ======= I. Process Data =======
        processed_data = self.process_data(data=data)
        series_mid = processed_data[0]
        series_high = processed_data[1]
        series_low = processed_data[2]

        # ======= II. Apply Smoothing if Needed =======
        if smoothing_method is not None:
            series_mid = self.smooth_data(data=series_mid, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_high = self.smooth_data(data=series_high, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)
            series_low = self.smooth_data(data=series_low, smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth)

        # ======= II. Ensure the window is not too large =======
        num_obs = len(series_high) - window
        if num_obs <= 0:
            raise ValueError(f"Window size {window} is too large for the given data length {len(series_high)}.")

        # ======= III. Initialize Output Arrays =======
        fast_oscillator_values = np.full(num_obs, np.nan)
        slow_oscillator_values = np.full(num_obs, np.nan)

        # ======== IV. Iterate Over Observations ========
        for i in range(num_obs):
            # IV.1 Extract Time Windows
            series_mid_window = series_mid.iloc[i : i + window]
            series_high_window = series_high.iloc[i : i + window]
            series_low_window = series_low.iloc[i : i + window]

            # IV.2 Perform Stochastic Oscillator Calculation
            fast_oscillator, slow_oscillator = utils.get_stochastic_oscillator(
                series_mid=series_mid_window,
                series_high=series_high_window, 
                series_low=series_low_window
            )

            # IV.3 Store Results
            fast_oscillator_values[i] = fast_oscillator
            slow_oscillator_values[i] = slow_oscillator

        # ======== V. Create the Final DataFrame ========
        index = series_high.index[window:]
        features_df = pd.DataFrame({
            f"{self.name}_fast_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": fast_oscillator_values,
            f"{self.name}_slow_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": slow_oscillator_values,
        }, index=index)

        return features_df

