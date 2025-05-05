from ..Measures import Filters as fil
from ..Measures import Momentum as mom
from ..Measures import Entropy as ent

from ..Features import common as com
from ...Model_Training.Models import linearRegression as reg

import numpy as np
import pandas as pd
from typing import Union

#! ==================================================================================== #
#! ======================= Unscaled Smoothed-like Series Features ===================== #
class average_feature(com.Feature):
    """
    Moving Average Feature

    This class computes the normalized moving average of a time series, with optional pre-smoothing filters.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving average feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "average" , 
        n_jobs: int = 1
    ):
        """
        Initializes the average_feature object with input data, name, and parallel jobs.
        
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
    ):
        """
        Defines the parameter grid for feature extraction.

        Parameters:
            - window (list): Rolling window sizes for the moving average.
            - smoothing_method (list): Type of pre-smoothing to apply. Options: None, "ewma", "average".
            - window_smooth (list): Window size for smoothing methods.
            - lambda_smooth (list): Smoothing factor for EWMA, in [0, 1].
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
    ):
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
        Computes the normalized rolling average of the processed series.

        Parameters: 
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the moving average.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_average (pd.Series): The resulting normalized moving average feature.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the moving average =======
        rolling_average = processed_series.rolling(window=window).apply(np.mean, raw=False)

        # ======= III. Convert to pd.Series and Center =======
        rolling_average = (pd.Series(rolling_average, index=processed_series.index) / (processed_series + 1e-8)) - 1
        
        # ======= IV. Change Name =======
        rolling_average.name = f"average_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_average

#*____________________________________________________________________________________ #
class median_feature(com.Feature):
    """
    Rolling Median Feature

    This class computes the normalized rolling median of a time series, with optional pre-smoothing filters.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving median feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "median", 
        n_jobs: int = 1
    ):
        """
        Initializes the median_feature object with input data, name, and parallel jobs.
        
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
    ):
        """
        Defines the parameter grid for feature extraction.

        Parameters:
            - window (list): Rolling window sizes for the median calculation.
            - smoothing_method (list): Type of pre-smoothing to apply. Options: None, "ewma", "average".
            - window_smooth (list): Window size for smoothing methods.
            - lambda_smooth (list): Smoothing factor for EWMA, in [0, 1].
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
    ):
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
        Computes the normalized rolling median of the processed series.

        Parameters: 
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the moving median.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_median (pd.Series): The resulting normalized rolling median feature.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        ).copy()
        
        processed_series = self.process_data(data=smoothed_series).dropna().copy()

        # ======= II. Compute the moving median =======
        rolling_median = processed_series.rolling(window=window).apply(np.median, raw=False)

        # ======= III. Convert to pd.Series and Center =======
        rolling_median = (pd.Series(rolling_median, index=processed_series.index) / (processed_series + 1e-8)) - 1
        
        # ======= IV. Change Name =======
        rolling_median.name = f"median_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_median

#*____________________________________________________________________________________ #
class minimum_feature(com.Feature):
    """
    Rolling Minimum Feature

    This class computes the normalized rolling minimum of a time series, with optional pre-smoothing filters.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving minimum feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "minimum" , 
        n_jobs: int = 1
    ):
        """
        Initializes the minimum_feature object with input data, name, and parallel jobs.

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
    ):
        """
        Defines the parameter grid for feature extraction.

        Parameters:
            - window (list): Rolling window sizes for the minimum calculation.
            - smoothing_method (list): Type of pre-smoothing to apply. Options: None, "ewma", "average".
            - window_smooth (list): Window size for smoothing methods.
            - lambda_smooth (list): Smoothing factor for EWMA, in [0, 1].
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
    ):
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
        Computes the normalized rolling minimum of the processed series.

        Parameters: 
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the moving minimum.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_min (pd.Series): The resulting normalized rolling minimum feature.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        ).copy()
        
        processed_series = self.process_data(data=smoothed_series).dropna().copy()

        # ======= II. Compute the moving minimum =======
        rolling_min = processed_series.rolling(window=window).apply(np.min, raw=False)

        # ======= II. Convert to pd.Series and Center =======
        rolling_min = (pd.Series(rolling_min, index=processed_series.index) / (processed_series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_min.name = f"min_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_min
    
#*____________________________________________________________________________________ #
class maximum_feature(com.Feature):
    """
    Rolling Maximum Feature

    This class computes the normalized rolling maximum of a time series, with optional pre-smoothing filters.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving maximum feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "maximum" , 
        n_jobs: int = 1
    ):
        """
        Initializes the maximum_feature object with input data, name, and parallel jobs.

        Parameters:
            - data (pd.Series): The time series data to be processed.
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
    ):
        """
        Defines the parameter grid for feature extraction.

        Parameters:
            - window (list): Rolling window sizes for the maximum calculation.
            - smoothing_method (list): Type of pre-smoothing to apply. Options: None, "ewma", "average".
            - window_smooth (list): Window size for smoothing methods.
            - lambda_smooth (list): Smoothing factor for EWMA, in [0, 1].
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
    ):
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
        Computes the normalized rolling maximum of the processed series.

        Parameters: 
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the moving maximum.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_max (pd.Series): The resulting normalized rolling maximum feature.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the moving maximum =======
        rolling_max = processed_series.rolling(window=window).apply(np.max, raw=False)

        # ======= II. Convert to pd.Series and Center =======
        rolling_max = (pd.Series(rolling_max, index=processed_series.index) / (processed_series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_max.name = f"max_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_max



#! ==================================================================================== #
#! ========================== Returns Distribution Features =========================== #
class volatility_feature(com.Feature):
    """
    Rolling Volatility Feature

    This class computes the rolling volatility (standard deviation of returns) of a time series,
    with optional smoothing filters applied beforehand.

    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving volatility feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "volatility" , 
        n_jobs: int = 1
    ):
        """
        Initializes the volatility_feature object with data, feature name, and parallel jobs.

        Parameters:
            - data (pd.Series): The input time series to compute volatility on.
            - name (str): Label for the feature, used in output series.
            - n_jobs (int): Number of parallel jobs for multi-core processing.
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
    ):
        """
        Defines the parameter grid for volatility feature extraction.

        Parameters:
            - window (list): Window sizes for computing rolling volatility.
            - smoothing_method (list): Type of pre-smoothing. Options: None, "ewma", or "average".
            - window_smooth (list): Smoothing window sizes for selected smoothing methods.
            - lambda_smooth (list): Smoothing decay factors for EWMA, values in [0, 1].
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
    ):
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
        Computes the rolling volatility (standard deviation of percentage returns).

        Parameters:
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the moving volatility.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_vol (pd.Series): The computed volatility feature as a time series.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the moving volatility =======
        returns_series = processed_series.pct_change().dropna()
        rolling_vol = returns_series.rolling(window=window).apply(np.std, raw=False)

        # ======= II. Convert to pd.Series and Center =======
        rolling_vol = pd.Series(rolling_vol, index=processed_series.index)
        
        # ======= III. Change Name =======
        rolling_vol.name = f"vol_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_vol

#*____________________________________________________________________________________ #
class skewness_feature(com.Feature):
    """
    Rolling Skewness Feature

    This class computes the rolling skewness of the return series derived from a time series,
    with optional pre-smoothing.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving skewness feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "skewness" , 
        n_jobs: int = 1
    ):
        """
        Initializes the skewness_feature object with the input series and parameters.

        Parameters:
            - data (pd.Series): The raw input time series.
            - name (str): Name of the feature.
            - n_jobs (int): Number of parallel jobs to use for processing.
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
    ):
        """
        Sets the parameter grid for skewness feature extraction.

        Parameters:
            - window (list): Rolling window sizes for skewness computation.
            - smoothing_method (list): Type of smoothing filter: "ewma", "average", or None.
            - window_smooth (list): Window sizes for smoothing methods.
            - lambda_smooth (list): Decay factors for EWMA smoothing, values in [0, 1].
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
    ):
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
        Computes the rolling skewness of percentage returns over a given window.

        Parameters:
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the moving skewness.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_skew (pd.Series): Time series of rolling skewness values.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the moving skewness =======
        returns_series = processed_series.pct_change().dropna()
        rolling_skew = returns_series.rolling(window=window).apply(lambda x: x.skew())

        # ======= II. Convert to pd.Series and Center =======
        rolling_skew = pd.Series(rolling_skew, index=processed_series.index)
        
        # ======= III. Change Name =======
        rolling_skew.name = f"skew_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_skew

#*____________________________________________________________________________________ #
class kurtosis_feature(com.Feature):
    """
    Rolling Kurtosis Feature

    This class computes the rolling kurtosis of the return series derived from a time series,
    with optional smoothing applied beforehand.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving kurtosis feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "kurtosis" , 
        n_jobs: int = 1
    ):
        """
        Initializes the kurtosis_feature object with the input series and basic config.

        Parameters:
            - data (pd.Series): The raw input time series.
            - name (str): Name of the feature.
            - n_jobs (int): Number of parallel jobs to use for processing.
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
    ):
        """
        Sets the parameter grid for kurtosis feature extraction.

        Parameters:
            - window (list): Rolling window sizes for kurtosis computation.
            - smoothing_method (list): Smoothing type: "ewma", "average", or None.
            - window_smooth (list): Window sizes for smoothing.
            - lambda_smooth (list): Decay factors for EWMA, values in [0, 1].
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
    ):
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
        Computes the rolling kurtosis of percentage returns over a given window.

        Parameters:
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the moving kurtosis.
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_kurt (pd.Series): Time series of rolling kurtosis values.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the moving kurtosis =======
        returns_series = processed_series.pct_change().dropna()
        rolling_kurt = returns_series.rolling(window=window).apply(lambda x: x.kurtosis())

        # ======= II. Convert to pd.Series and Center =======
        rolling_kurt = pd.Series(rolling_kurt, index=processed_series.index)
        
        # ======= III. Change Name =======
        rolling_kurt.name = f"kurt_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_kurt

#*____________________________________________________________________________________ #
class quantile_feature(com.Feature):
    """
    Rolling Quantile Feature

    This class computes the rolling quantile of the return series derived from a time series,
    with optional smoothing applied beforehand.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving quantile feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "quantile" , 
        n_jobs: int = 1
    ):
        """
        Initializes the quantile_feature object with the input series and basic config.

        Parameters:
            - data (pd.Series): The raw input time series.
            - name (str): Name of the feature.
            - n_jobs (int): Number of parallel jobs to use for processing.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
        )
        
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        window: list = [5, 10, 30, 60],
        quantile: list = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99],
        smoothing_method: list = [None, "ewma", "average"],
        window_smooth: list = [5, 10],
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ):
        """
        Sets the parameter grid for quantile feature extraction.

        Parameters:
            - window (list): Rolling window sizes for quantile computation.
            - quantile (list): Quantile levels to compute, must be in [0, 1].
            - smoothing_method (list): Smoothing method to apply before computation.
            - window_smooth (list): Smoothing window sizes.
            - lambda_smooth (list): Decay factors for EWMA, values in [0, 1].
        """
        self.params = {
            "window": window,
            "quantile": quantile,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }

        return self

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: pd.Series,
    ):
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
        quantile: float,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the rolling quantile of percentage returns over a specified window.

        Parameters:
            - data (pd.Series): The input series to be processed.
            - window (int): Rolling window size for the moving quantile.
            - quantile (float): Quantile level to compute, must be in [0, 1].
            - smoothing_method (str): Smoothing method used.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_quantile (pd.Series): Time series of rolling quantile values.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the moving quantile =======
        returns_series = processed_series.pct_change().dropna()
        rolling_quantile = returns_series.rolling(window=window).apply(lambda x: np.quantile(x, quantile))

        # ======= II. Convert to pd.Series and Center =======
        rolling_quantile = pd.Series(rolling_quantile, index=processed_series.index)
        
        # ======= III. Change Name =======
        rolling_quantile.name = f"quantile_{quantile}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_quantile



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
    ):
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
    ):
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
    ):
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
        rolling_momentum = processed_series.rolling(window=window ).apply(mom.get_momentum, raw=False)
        
        # ======= III. Convert to pd.Series and Center =======
        rolling_momentum = pd.Series(rolling_momentum, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_momentum.name = f"momentum_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

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
    ):
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
    ):
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
    ):
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
        rolling_Z_momentum = processed_series.rolling(window=window).apply(mom.get_Z_momentum, raw=False)
        
        # ======= III. Convert to pd.Series and Center =======
        rolling_Z_momentum = pd.Series(rolling_Z_momentum, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_Z_momentum.name = f"Z_momentum_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_Z_momentum

#*____________________________________________________________________________________ #
class linear_tempReg_feature(com.Feature):
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
    ):
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
    ):
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
    ):
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
        def compute_slope(series):
            _, coefficients, _, _ = mom.get_simple_TempReg(series)
            slope = coefficients[0]
            
            return slope

        def compute_T_stats(series):
            _, _, statistics, _ = mom.get_simple_TempReg(series)
            T_stats = statistics['T_stats'][0]
            
            return T_stats
        
        def compute_Pvalue(series):
            _, _, statistics, _ = mom.get_simple_TempReg(series)
            P_value = statistics['P_values'][0]
            
            return P_value
        
        def compute_R_squared(series):
            _, _, statistics, _ = mom.get_simple_TempReg(series)
            R_squared = statistics['R_squared']
            
            return R_squared

        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the rolling regression statistics =======
        rolling_slope = processed_series.rolling(window=window).apply(compute_slope, raw=False)
        rolling_tstat = processed_series.rolling(window=window).apply(compute_T_stats, raw=False)
        rolling_pvalue = processed_series.rolling(window=window).apply(compute_Pvalue, raw=False)
        rolling_r_squared = processed_series.rolling(window=window).apply(compute_R_squared, raw=False)

        # ======= III. Convert to pd.Series and Unscale =======
        rolling_slope = pd.Series(rolling_slope, index=processed_series.index) / (processed_series + 1e-8)
        rolling_tstat = pd.Series(rolling_tstat, index=processed_series.index)
        rolling_pvalue = pd.Series(rolling_pvalue, index=processed_series.index)
        rolling_r_squared = pd.Series(rolling_r_squared, index=processed_series.index)
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"linear_slope_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_slope,
            f"linear_tstat_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_tstat,
            f"linear_pvalue_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_pvalue,
            f"linear_r_squared_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_r_squared,
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
    ):
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
    ):
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
    ):
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
        def compute_slope(series):
            _, coefficients, _, _ = mom.get_quad_TempReg(series)
            slope = coefficients[0]
            
            return slope
        
        def compute_acceleration(series):
            _, coefficients, _, _ = mom.get_quad_TempReg(series)
            acceleration = coefficients[1]
            
            return acceleration

        def compute_T_stats(series):
            _, _, statistics, _ = mom.get_quad_TempReg(series)
            T_stats = statistics['T_stats'][0]
            
            return T_stats
        
        def compute_Pvalue(series):
            _, _, statistics, _ = mom.get_quad_TempReg(series)
            P_value = statistics['P_values'][0]
            
            return P_value
        
        def compute_R_squared(series):
            _, _, statistics, _ = mom.get_quad_TempReg(series)
            R_squared = statistics['R_squared']
            
            return R_squared

        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the rolling regression statistics =======
        rolling_slope = processed_series.rolling(window=window).apply(compute_slope, raw=False)
        rolling_acceleration = processed_series.rolling(window=window).apply(compute_acceleration, raw=False)
        rolling_tstat = processed_series.rolling(window=window).apply(compute_T_stats, raw=False)
        rolling_pvalue = processed_series.rolling(window=window).apply(compute_Pvalue, raw=False)
        rolling_r_squared = processed_series.rolling(window=window).apply(compute_R_squared, raw=False)

        # ======= III. Convert to pd.Series and Unscale =======
        rolling_slope = pd.Series(rolling_slope, index=processed_series.index) / (processed_series + 1e-8)
        rolling_acceleration = pd.Series(rolling_acceleration, index=processed_series.index) / (processed_series + 1e-8)
        rolling_tstat = pd.Series(rolling_tstat, index=processed_series.index)
        rolling_pvalue = pd.Series(rolling_pvalue, index=processed_series.index)
        rolling_r_squared = pd.Series(rolling_r_squared, index=processed_series.index)
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"nonlinear_slope_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_slope,
            f"nonlinear_acceleration_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_acceleration,
            f"nonlinear_tstat_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_tstat,
            f"nonlinear_pvalue_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_pvalue,
            f"nonlinear_r_squared_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_r_squared,
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
        name: str = "hurst_exponent" , 
        n_jobs: int = 1
    ):
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
    ):
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
    ):
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
    ):
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

            model = reg.OLSRegression()
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
            f"hurst_exponent{power}_{smoothing_method}_{window_smooth}_{lambda_smooth}": hursts,
            f"hurst_tstat_{power}_{smoothing_method}_{window_smooth}_{lambda_smooth}": tstats,
            f"hurst_pvalue_{power}_{smoothing_method}_{window_smooth}_{lambda_smooth}": pvalues,
        })
        
        return features_df



#! ==================================================================================== #
#! ============================= Signal Processing Features =========================== #
class entropy_feature(com.Feature):
    """
    Rolling Entropy Feature Extraction

    This class computes various entropy-based measures (Shannon, Plugin, Lempel-Ziv, and Kontoyiannis)
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving entropies feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "entropy" , 
        n_jobs: int = 1
    ):
        """
        Initializes the entropy_feature object with the input series.

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
    ):
        """
        Sets the parameter grid for entropy feature extraction.

        Parameters:
            - window (list): Rolling window sizes for entropy computation.
            - smoothing_method (list): Type of smoothing to apply before entropy calculation.
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
    ):
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
        Computes rolling entropy features from the smoothed series.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - window (int): Rolling window size for entropy calculation.
            - smoothing_method (str): Smoothing method to apply prior to entropy calculation.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - features_df (pd.DataFrame): DataFrame with Shannon, Plugin, 
              Lempel-Ziv, and Kontoyiannis entropy values for each window.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()
        
        # ======= II. Compute the rolling entropy features =======
        signs_series = ent.get_movements_signs(series=processed_series)
        rolling_shannon = signs_series.rolling(window=window).apply(ent.get_shannon_entropy, raw=False)
        rolling_plugin = signs_series.rolling(window=window).apply(ent.get_plugin_entropy, raw=False)
        rolling_lempel_ziv = signs_series.rolling(window=window).apply(ent.get_lempel_ziv_entropy, raw=False)
        rolling_kontoyiannis = signs_series.rolling(window=window).apply(ent.get_kontoyiannis_entropy, raw=False)

        # ======= III. Convert to pd.Series and Center =======
        rolling_shannon = pd.Series(rolling_shannon, index=processed_series.index)
        rolling_plugin = pd.Series(rolling_plugin, index=processed_series.index)
        rolling_lempel_ziv = pd.Series(rolling_lempel_ziv, index=processed_series.index)
        rolling_kontoyiannis = pd.Series(rolling_kontoyiannis, index=processed_series.index)
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"shannon_entropy_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_shannon,
            f"plugin_entropy_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_plugin,
            f"lempel_ziv_entropy_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_lempel_ziv,
            f"kontoyiannis_entropy_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_kontoyiannis,
        })

        return features_df

#*____________________________________________________________________________________ #
class sample_entropy_feature(com.Feature):
    """
    Sample Entropy Feature

    This class computes the sample entropy over a rolling window of a time series.
    It inherits from the Feature base class and implements methods to:
        - define parameter grids
        - apply optional preprocessing
        - compute sample entropy feature
    """
    def __init__(
        self, 
        name: str = "sample_entropy", 
        n_jobs: int = 1
    ):
        """
        Initializes the sample_entropy_feature object with input data, name, and parallel jobs.
        
        Parameters:
            - data (pd.Series): The time series data to be processed.
            - name (str): Name of the feature, used in column labeling.
            - n_jobs (int): Number of jobs to run in parallel for feature extraction.
        """
        super().__init__(
            name=name, 
            n_jobs=n_jobs
        )

    #?____________________________________________________________________________________ #
    def set_params(
        self,
        window: list = [5, 10, 30, 60],
        sub_vector_size: list = [2, 3],
        threshold_distance: list = [0.1, 0.2, 0.3],
        smoothing_method: list = [None, "ewma", "average"],
        window_smooth: list = [5, 10],
        lambda_smooth: list = [0.1, 0.2, 0.5],
    ):
        """
        Defines the parameter grid for feature extraction.

        Parameters:
            - window (list): Rolling window sizes for sample entropy.
            - sub_vector_size (list): Embedding dimension values.
            - threshold_distance (list): Tolerance values as a fraction of standard deviation.
        """
        self.params = {
            "window": window,
            "sub_vector_size": sub_vector_size,
            "threshold_distance": threshold_distance,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }

        return self

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: pd.Series,
    ):
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
        sub_vector_size: int,
        threshold_distance: float,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the rolling sample entropy over the processed series.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - window (int): Rolling window size for sample entropy.
            - sub_vector_size (int): Embedding dimension.
            - threshold_distance (float): Tolerance for entropy, as a fraction of std.

        Returns:
            - sample_entropy_series (pd.Series): Series of sample entropy values.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()
        
        # ======= II. Compute the rolling entropy features =======
        signs_series = ent.get_movements_signs(series=processed_series)
        rolling_entropy = signs_series.rolling(window=window).apply(lambda x: ent.calculate_sample_entropy(series=x, sub_vector_size=sub_vector_size, threshold_distance=threshold_distance), raw=False)

        # ======= III. Convert to pd.Series and Center =======
        rolling_entropy = pd.Series(rolling_entropy, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_entropy.name = f"sample_entropy_{sub_vector_size}_{threshold_distance}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        
        return rolling_entropy

