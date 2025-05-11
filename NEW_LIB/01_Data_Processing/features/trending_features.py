from ..features import common as com
from ...utils import calculations as calc

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Self



#! ==================================================================================== #
#! ============================= Series Trending Features ============================= #
class momentum_feature(com.Feature):
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
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the moving momentum =======
        rolling_momentum = processed_series.rolling(window=window ).apply(calc.get_momentum, raw=False)
        
        # ======= III. Convert to pd.Series and Center =======
        rolling_momentum = pd.Series(rolling_momentum, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_momentum.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

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
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the moving Z-momentum =======
        rolling_Z_momentum = processed_series.rolling(window=window).apply(calc.get_Z_momentum, raw=False)
        
        # ======= III. Convert to pd.Series and Center =======
        rolling_Z_momentum = pd.Series(rolling_Z_momentum, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_Z_momentum.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_Z_momentum

#*____________________________________________________________________________________ #
class linear_tempReg_feature():
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
            intercept, coefficients, statistics, residuals = calc.get_simple_TempReg(series=current_window)
            
            return start_idx, intercept, coefficients, statistics, residuals
            

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
        rolling_slope = pd.Series({i: coeffs[0] for i, _, coeffs, _, _ in results}) 
        rolling_tstat = pd.Series({i: stats['t_stats'][0] for i, _, _, stats, _ in results})
        rolling_pvalue = pd.Series({i: stats['p_values'][0] for i, _, _, stats, _ in results})
        rolling_r2 = pd.Series({i: stats['r2'] for i, _, _, stats, _ in results})
        
        # ======= IV. Rearrange the index =======
        rolling_slope.index = processed_series.index[window - 1:]
        rolling_tstat.index = processed_series.index[window - 1:]
        rolling_pvalue.index = processed_series.index[window - 1:]
        rolling_r2.index = processed_series.index[window - 1:]
        
        # ======= V. Center =======
        rolling_slope = rolling_slope / (processed_series.loc[rolling_slope.index] + 1e-8)
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"{self.name}_slope_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_slope,
            f"{self.name}_tstat_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_tstat,
            f"{self.name}_pvalue_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_pvalue,
            f"{self.name}_r2_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_r2,
        })
        
        return features_df

#*____________________________________________________________________________________ #
class nonlinear_tempReg_feature(com.Feature):
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
            intercept, coefficients, statistics, residuals = calc.get_quad_TempReg(series=current_window)
            
            return start_idx, intercept, coefficients, statistics, residuals
            

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
        rolling_slope = pd.Series({i: coeffs[0] for i, _, coeffs, _, _ in results}) 
        rolling_acceleration = pd.Series({i: coeffs[1] for i, _, coeffs, _, _ in results})
        rolling_tstat = pd.Series({i: stats['t_stats'][0] for i, _, _, stats, _ in results})
        rolling_pvalue = pd.Series({i: stats['p_values'][0] for i, _, _, stats, _ in results})
        rolling_r2 = pd.Series({i: stats['r2'] for i, _, _, stats, _ in results})
        
        # ======= IV. Rearrange the index =======
        rolling_slope.index = processed_series.index[window - 1:]
        rolling_acceleration.index = processed_series.index[window - 1:]
        rolling_tstat.index = processed_series.index[window - 1:]
        rolling_pvalue.index = processed_series.index[window - 1:]
        rolling_r2.index = processed_series.index[window - 1:]
        
        # ======= V. Center =======
        rolling_slope = rolling_slope / (processed_series.loc[rolling_slope.index] + 1e-8)
        rolling_acceleration = rolling_acceleration / (processed_series.loc[rolling_slope.index] + 1e-8)
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"{self.name}_slope_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_slope,
            f"{self.name}_acceleration_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_acceleration,
            f"{self.name}_tstat_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_tstat,
            f"{self.name}_pvalue_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_pvalue,
            f"{self.name}_r2_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_r2,
        })

        return features_df

#*____________________________________________________________________________________ #
class hurst_exponent_feature(com.Feature):
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

            model = calc.OLSRegression()
            model.fit(X, Y)
            
            hurst = model.coefficients[0]
            tstat = model.statistics['T_stats'][0]
            pvalue = model.statistics['P_values'][0]
            
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


