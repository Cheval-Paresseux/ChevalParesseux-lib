from ..Measures import Filters as fil
from ..Measures import Momentum as mom
from ..Measures import Entropy as ent

from ..Features import common as com
from ...Model_Training.Models import linearRegression as reg

import numpy as np
import pandas as pd

#! ==================================================================================== #
#! ======================= Unscaled Smoothed-like Series Features ===================== #
class average_feature(com.Feature):
    """
    Moving Average Feature

    This class computes the normalized moving average of a time series, with optional pre-smoothing filters.
    It inherits from the Feature base class and implements methods to:
        - set parameter grids
        - optionally smooth the input series
        - compute the moving average feature over a rolling window
    """
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        n_jobs: int = 1
    ):
        """
        Initializes the average_feature object with input data, name, and parallel jobs.
        
        Parameters:
            - data (pd.Series): The time series data to be processed.
            - name (str): Name of the feature, used in column labeling.
            - n_jobs (int): Number of jobs to run in parallel for feature extraction.
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies optional smoothing to the input data before feature computation.

        Parameters:
            - smoothing_method (str): Type of smoothing to apply. Options: "ewma", "average", or None.
            - window_smooth (int): Size of the smoothing window.
            - lambda_smooth (float): EWMA decay parameter in [0, 1].

        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the normalized rolling average of the processed series.

        Parameters: 
            - window (int): Rolling window size for the moving average.
            - smoothing_method (str): Smoothing method used in preprocessing.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_average (pd.Series): The resulting normalized moving average feature.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_average = series.rolling(window=window + 1).apply(lambda x: np.mean(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_average = (pd.Series(rolling_average, index=series.index) / (series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_average.name = f"average_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_average


#*____________________________________________________________________________________ #
class median_feature(com.Feature):
    """
    Rolling Median Feature

    This class computes the normalized rolling median of a time series, with optional pre-smoothing filters.
    It inherits from the Feature base class and implements methods to:
        - set parameter grids
        - optionally smooth the input series
        - compute the rolling median feature over a sliding window
    """
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "median" , 
        n_jobs: int = 1
    ):
        """
        Initializes the median_feature object with input data, name, and parallel jobs.
        
        Parameters:
            - data (pd.Series): The time series data to be processed.
            - name (str): Name of the feature, used in column labeling.
            - n_jobs (int): Number of jobs to run in parallel for feature extraction.
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies optional smoothing to the input data before feature computation.

        Parameters:
            - smoothing_method (str): Type of smoothing to apply. Options: "ewma", "average", or None.
            - window_smooth (int): Size of the smoothing window.
            - lambda_smooth (float): EWMA decay parameter in [0, 1].

        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the normalized rolling median of the processed series.

        Parameters: 
            - window (int): Rolling window size for the median computation.
            - smoothing_method (str): Smoothing method used in preprocessing.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_median (pd.Series): The resulting normalized rolling median feature.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_median = series.rolling(window=window + 1).apply(lambda x: np.median(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_median = (pd.Series(rolling_median, index=series.index) / (series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_median.name = f"median_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_median


#*____________________________________________________________________________________ #
class minimum_feature(com.Feature):
    """
    Rolling Minimum Feature

    This class computes the normalized rolling minimum of a time series, with optional pre-smoothing filters.
    It inherits from the Feature base class and implements methods to:
        - set parameter grids
        - optionally smooth the input series
        - compute the rolling minimum feature over a sliding window
    """
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "minimum" , 
        n_jobs: int = 1
    ):
        """
        Initializes the minimum_feature object with input data, name, and parallel jobs.

        Parameters:
            - data (pd.Series): The time series data to be processed.
            - name (str): Name of the feature, used in column labeling.
            - n_jobs (int): Number of jobs to run in parallel for feature extraction.
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies optional smoothing to the input data before feature computation.

        Parameters:
            - smoothing_method (str): Type of smoothing to apply. Options: "ewma", "average", or None.
            - window_smooth (int): Size of the smoothing window.
            - lambda_smooth (float): EWMA decay parameter in [0, 1].

        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the normalized rolling minimum of the processed series.

        Parameters: 
            - window (int): Rolling window size for the minimum computation.
            - smoothing_method (str): Smoothing method used in preprocessing.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_min (pd.Series): The resulting normalized rolling minimum feature.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_min = series.rolling(window=window + 1).apply(lambda x: np.min(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_min = (pd.Series(rolling_min, index=series.index) / (series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_min.name = f"min_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_min
    
    
#*____________________________________________________________________________________ #
class maximum_feature(com.Feature):
    """
    Rolling Maximum Feature

    This class computes the normalized rolling maximum of a time series, with optional pre-smoothing filters.
    It inherits from the Feature base class and implements methods to:
        - set parameter grids
        - optionally smooth the input series
        - compute the rolling maximum feature over a sliding window
    """
    def __init__(
        self, 
        data: pd.Series, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies optional smoothing to the input data before feature computation.

        Parameters:
            - smoothing_method (str): Type of smoothing to apply. Options: "ewma", "average", or None.
            - window_smooth (int): Size of the smoothing window.
            - lambda_smooth (float): EWMA decay parameter in [0, 1].

        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the normalized rolling maximum of the processed series.

        Parameters: 
            - window (int): Rolling window size for the maximum computation.
            - smoothing_method (str): Smoothing method used in preprocessing.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Smoothing parameter for EWMA.

        Returns:
            - rolling_max (pd.Series): The resulting normalized rolling maximum feature.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_max = series.rolling(window=window + 1).apply(lambda x: np.max(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_max = (pd.Series(rolling_max, index=series.index) / (series + 1e-8)) - 1
        
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
        - define parameter grids for feature extraction
        - smooth the original time series before processing
        - compute rolling volatility over a defined window
    """
    def __init__(
        self, 
        data: pd.Series, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies optional smoothing to the input time series.

        Parameters:
            - smoothing_method (str): Type of smoothing: "ewma", "average", or None.
            - window_smooth (int): Size of the rolling window for smoothing.
            - lambda_smooth (float): Decay factor for EWMA, in [0, 1].

        Returns:
            - processed_data (pd.Series): Smoothed or raw time series depending on parameters.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the rolling volatility (standard deviation of percentage returns).

        Parameters:
            - window (int): Size of the rolling window for volatility computation.
            - smoothing_method (str): Type of pre-smoothing filter used.
            - window_smooth (int): Window size used in smoothing.
            - lambda_smooth (float): EWMA decay factor if applicable.

        Returns:
            - rolling_vol (pd.Series): The computed volatility feature as a time series.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        returns_series = series.pct_change().dropna()
        rolling_vol = returns_series.rolling(window=window + 1).apply(lambda x: np.std(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_vol = pd.Series(rolling_vol, index=series.index)
        
        # ======= III. Change Name =======
        rolling_vol.name = f"vol_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_vol


#*____________________________________________________________________________________ #
class skewness_feature(com.Feature):
    """
    Rolling Skewness Feature

    This class computes the rolling skewness of the return series derived from a time series,
    with optional pre-smoothing.

    Inherits from the Feature base class and provides:
        - definition of parameter grids
        - data preprocessing through smoothing
        - rolling skewness feature computation
    """
    def __init__(
        self, 
        data: pd.Series, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies smoothing to the input series before computing skewness.

        Parameters:
            - smoothing_method (str): Smoothing type. Options: "ewma", "average", or None.
            - window_smooth (int): Smoothing window size (in number of bars).
            - lambda_smooth (float): Decay factor for EWMA, must be in [0, 1].

        Returns:
            - processed_data (pd.Series): Smoothed time series (or raw if no smoothing).
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the rolling skewness of percentage returns over a given window.

        Parameters:
            - window (int): Size of the rolling window used to compute skewness.
            - smoothing_method (str): Pre-smoothing method applied to the series.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): EWMA decay factor if applicable.

        Returns:
            - rolling_skew (pd.Series): Time series of rolling skewness values.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        returns_series = series.pct_change().dropna()
        rolling_skew = returns_series.rolling(window=window + 1).apply(lambda x: (x[:window]).skew())

        # ======= II. Convert to pd.Series and Center =======
        rolling_skew = pd.Series(rolling_skew, index=series.index)
        
        # ======= III. Change Name =======
        rolling_skew.name = f"skew_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_skew


#*____________________________________________________________________________________ #
class kurtosis_feature(com.Feature):
    """
    Rolling Kurtosis Feature

    This class computes the rolling kurtosis of the return series derived from a time series,
    with optional smoothing applied beforehand.

    Inherits from the Feature base class and provides:
        - definition of parameter grids
        - preprocessing of the data (smoothing)
        - computation of the rolling kurtosis
    """
    def __init__(
        self, 
        data: pd.Series, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies smoothing to the input series before computing kurtosis.

        Parameters:
            - smoothing_method (str): Smoothing type. Options: "ewma", "average", or None.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA (range [0, 1]).

        Returns:
            - processed_data (pd.Series): Smoothed time series.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the rolling kurtosis of percentage returns over a given window.

        Parameters:
            - window (int): Size of the rolling window.
            - smoothing_method (str): Type of smoothing applied before calculation.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA smoothing.

        Returns:
            - rolling_kurt (pd.Series): Time series of rolling kurtosis values.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        returns_series = series.pct_change().dropna()
        rolling_kurt = returns_series.rolling(window=window + 1).apply(lambda x: (x[:window]).kurtosis())

        # ======= II. Convert to pd.Series and Center =======
        rolling_kurt = pd.Series(rolling_kurt, index=series.index)
        
        # ======= III. Change Name =======
        rolling_kurt.name = f"kurt_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_kurt


#*____________________________________________________________________________________ #
class quantile_feature(com.Feature):
    """
    Rolling Quantile Feature

    This class computes the rolling quantile of the return series derived from a time series,
    with optional smoothing applied beforehand.

    Inherits from the Feature base class and provides:
        - definition of parameter grids
        - preprocessing of the data (smoothing)
        - computation of the rolling quantile
    """
    def __init__(
        self, 
        data: pd.Series, 
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
            data=data, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies smoothing to the input series before computing quantile.

        Parameters:
            - smoothing_method (str): Type of smoothing. Options: "ewma", "average", or None.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA (range [0, 1]).

        Returns:
            - processed_data (pd.Series): Smoothed time series.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        quantile: float,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the rolling quantile of percentage returns over a specified window.

        Parameters:
            - window (int): Size of the rolling window.
            - quantile (float): Quantile level to compute, must be in [0, 1].
            - smoothing_method (str): Type of smoothing applied before calculation.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA smoothing.

        Returns:
            - rolling_quantile (pd.Series): Time series of rolling quantile values.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        returns_series = series.pct_change().dropna()
        rolling_quantile = returns_series.rolling(window=window + 1).apply(lambda x: np.quantile(x[:window], quantile))

        # ======= II. Convert to pd.Series and Center =======
        rolling_quantile = pd.Series(rolling_quantile, index=series.index)
        
        # ======= III. Change Name =======
        rolling_quantile.name = f"quantile_{quantile}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_quantile



#! ==================================================================================== #
#! ============================= Series Trending Features ============================= #
class momentum_feature(com.Feature):
    """
    Rolling Momentum Feature

    This class computes the rolling momentum of a time series.
    Momentum is calculated using a custom function (mom.get_momentum),
    with optional smoothing applied to the series before the computation.

    Inherits from the Feature base class.
    """
    def __init__(
        self, 
        data: pd.Series, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies a smoothing filter to the series prior to feature calculation.

        Parameters:
            - smoothing_method (str): Options are "ewma", "average", or None.
            - window_smooth (int): Window size for smoothing.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - processed_data (pd.Series): Smoothed series.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the rolling momentum from the smoothed series.

        Parameters:
            - window (int): Rolling window size.
            - smoothing_method (str): Smoothing method to apply.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - rolling_momentum (pd.Series): Series of rolling momentum values.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_momentum = series.rolling(window=window + 1).apply(lambda x: mom.get_momentum(x[:window]))
        
        # ======= II. Convert to pd.Series and Center =======
        rolling_momentum = pd.Series(rolling_momentum, index=series.index)
        
        # ======= III. Change Name =======
        rolling_momentum.name = f"momentum_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_momentum


#*____________________________________________________________________________________ #
class Z_momentum_feature(com.Feature):
    """
    Rolling Z-Momentum Feature

    This class computes the rolling Z-momentum of a time series.
    Z-momentum is a normalized momentum value (e.g., z-score of returns or momentum),
    useful for comparing across assets or regimes.

    Inherits from the Feature base class.
    """
    def __init__(
        self, 
        data: pd.Series, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies a smoothing filter to the series prior to feature calculation.

        Parameters:
            - smoothing_method (str): Options are "ewma", "average", or None.
            - window_smooth (int): Window size for smoothing.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - processed_data (pd.Series): Smoothed series.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the rolling Z-momentum from the smoothed series.

        Parameters:
            - window (int): Rolling window size.
            - smoothing_method (str): Smoothing method to apply.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - rolling_Z_momentum (pd.Series): Series of rolling Z-momentum values.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_Z_momentum = series.rolling(window=window + 1).apply(lambda x: mom.get_Z_momentum(x[:window]))
        
        # ======= II. Convert to pd.Series and Center =======
        rolling_Z_momentum = pd.Series(rolling_Z_momentum, index=series.index)
        
        # ======= III. Change Name =======
        rolling_Z_momentum.name = f"Z_momentum_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_Z_momentum


#*____________________________________________________________________________________ #
class linear_tempReg_feature(com.Feature):
    """
    Rolling Linear Regression Feature

    This class computes rolling linear regression statistics over a time series. 
    For each window, it fits a linear model and extracts the slope, t-statistic, 
    p-value, and R-squared. These metrics capture local trends and the statistical 
    strength of the fitted trend.

    Smoothing options such as EWMA or simple average can be applied to the series 
    before regression to reduce noise and improve signal quality.

    Inherits from the Feature base class.
    """
    def __init__(
        self, 
        data: pd.Series, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies a smoothing filter to the input series prior to regression analysis.

        Parameters:
            - smoothing_method (str): Options are "ewma", "average", or None.
            - window_smooth (int): Window size for smoothing.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - processed_data (pd.Series): Smoothed time series.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes rolling linear regression statistics (slope, t-stat, p-value, R-squared)
        on the smoothed series over the specified window.

        Parameters:
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

        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()

        # ======= II. Compute the rolling regression statistics =======
        rolling_slope = series.rolling(window=window + 1).apply(compute_slope, raw=False)
        rolling_tstat = series.rolling(window=window + 1).apply(compute_T_stats, raw=False)
        rolling_pvalue = series.rolling(window=window + 1).apply(compute_Pvalue, raw=False)
        rolling_r_squared = series.rolling(window=window + 1).apply(compute_R_squared, raw=False)

        # ======= III. Convert to pd.Series and Unscale =======
        rolling_slope = pd.Series(rolling_slope, index=series.index) / (series + 1e-8)
        rolling_tstat = pd.Series(rolling_tstat, index=series.index)
        rolling_pvalue = pd.Series(rolling_pvalue, index=series.index)
        rolling_r_squared = pd.Series(rolling_r_squared, index=series.index)
        
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
    Rolling Nonlinear (Quadratic) Regression Feature

    This class computes nonlinear regression statistics—slope (linear term), 
    acceleration (quadratic term), t-statistic, p-value, and R-squared—over rolling 
    windows of a time series. Useful for identifying curvature and more complex 
    trend dynamics than linear regression alone.

    Inherits from the Feature base class.
    """
    def __init__(
        self, 
        data: pd.Series, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies a smoothing filter to the series prior to feature calculation.

        Parameters:
            - smoothing_method (str): Options are "ewma", "average", or None.
            - window_smooth (int): Window size for smoothing.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - processed_data (pd.Series): Smoothed series.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes rolling nonlinear regression features from the smoothed series.

        Parameters:
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

        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()

        # ======= II. Compute the rolling regression statistics =======
        rolling_slope = series.rolling(window=window + 1).apply(compute_slope, raw=False)
        rolling_acceleration = series.rolling(window=window + 1).apply(compute_acceleration, raw=False)
        rolling_tstat = series.rolling(window=window + 1).apply(compute_T_stats, raw=False)
        rolling_pvalue = series.rolling(window=window + 1).apply(compute_Pvalue, raw=False)
        rolling_r_squared = series.rolling(window=window + 1).apply(compute_R_squared, raw=False)

        # ======= III. Convert to pd.Series and Unscale =======
        rolling_slope = pd.Series(rolling_slope, index=series.index) / (series + 1e-8)
        rolling_acceleration = pd.Series(rolling_acceleration, index=series.index) / (series + 1e-8)
        rolling_tstat = pd.Series(rolling_tstat, index=series.index)
        rolling_pvalue = pd.Series(rolling_pvalue, index=series.index)
        rolling_r_squared = pd.Series(rolling_r_squared, index=series.index)
        
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

    This class computes the Hurst exponent, its associated t-statistic, and p-value 
    over rolling windows derived from log returns. The Hurst exponent indicates 
    the degree of long-term memory of a time series, where values > 0.5 suggest 
    trend-following behavior and values < 0.5 suggest mean-reversion.

    The output series is optionally smoothed before analysis to improve signal 
    quality and reduce noise.

    Inherits from the Feature base class.
    """
    def __init__(
        self, 
        data: pd.Series, 
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
            data=data, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies a smoothing filter to the series prior to feature calculation.

        Parameters:
            - smoothing_method (str): Options are "ewma", "average", or None.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - processed_data (pd.Series): Smoothed time series.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        power: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes rolling Hurst exponent values, along with t-statistics and p-values.

        Parameters:
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
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        prices_array = np.array(series)
        returns_array = prices_array[1:] / prices_array[:-1] - 1

        n = 2**power

        hursts = np.array([])
        tstats = np.array([])
        pvalues = np.array([])

        # ======= II. Compute the Hurst Exponent =======
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
        hursts = pd.Series([np.nan] * n + list(hursts), index=series.index) - 0.5
        tstats = pd.Series([np.nan] * n + list(tstats), index=series.index)
        pvalues = pd.Series([np.nan] * n + list(pvalues), index=series.index)

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

    This class computes various entropy-based measures—Shannon, Plugin, Lempel-Ziv, 
    and Kontoyiannis—over rolling windows of a time series. It is designed to 
    capture the randomness, complexity, and compressibility of the series, 
    especially after transformation into sign sequences.

    Inherits from the Feature base class.
    """
    def __init__(
        self, 
        data: pd.Series, 
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
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies a smoothing filter to the series prior to entropy calculation.

        Parameters:
            - smoothing_method (str): Options are "ewma", "average", or None.
            - window_smooth (int): Window size for smoothing.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - processed_data (pd.Series): Smoothed series.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            processed_data = self.data
            self.processed_data = processed_data
            return processed_data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        # ======= III. Save the processed data =======
        self.processed_data = processed_data
        
        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes rolling entropy features from the smoothed series.

        Parameters:
            - window (int): Rolling window size for entropy calculation.
            - smoothing_method (str): Smoothing method to apply prior to entropy calculation.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - features_df (pd.DataFrame): DataFrame with Shannon, Plugin, 
              Lempel-Ziv, and Kontoyiannis entropy values for each window.
        """
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        signs_series = ent.get_movements_signs(series=series)

        # ======= II. Compute the rolling entropy features =======
        rolling_shannon = signs_series.rolling(window=window + 1).apply(ent.get_shannon_entropy, raw=False)
        rolling_plugin = signs_series.rolling(window=window + 1).apply(ent.get_plugin_entropy, raw=False)
        rolling_lempel_ziv = signs_series.rolling(window=window + 1).apply(ent.get_lempel_ziv_entropy, raw=False)
        rolling_kontoyiannis = signs_series.rolling(window=window + 1).apply(ent.get_kontoyiannis_entropy, raw=False)

        # ======= III. Convert to pd.Series and Center =======
        rolling_shannon = pd.Series(rolling_shannon, index=series.index)
        rolling_plugin = pd.Series(rolling_plugin, index=series.index)
        rolling_lempel_ziv = pd.Series(rolling_lempel_ziv, index=series.index)
        rolling_kontoyiannis = pd.Series(rolling_kontoyiannis, index=series.index)
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"shannon_entropy_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_shannon,
            f"plugin_entropy_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_plugin,
            f"lempel_ziv_entropy_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_lempel_ziv,
            f"kontoyiannis_entropy_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_kontoyiannis,
        })

        return features_df


