from ..features import common as com

import numpy as np
import pandas as pd
from typing import Union, Self



#! ==================================================================================== #
#! ================================= Center Features ================================== #
class Average_feature(com.Feature):
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
    ) -> None:
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
    ) -> Self:
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
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the moving average =======
        rolling_average = processed_series.rolling(window=window).apply(np.mean, raw=False)

        # ======= III. Convert to pd.Series and Center =======
        rolling_average = (pd.Series(rolling_average, index=processed_series.index) / (processed_series + 1e-8)) - 1
        
        # ======= IV. Change Name =======
        rolling_average.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_average.index = data.index

        return rolling_average

#*____________________________________________________________________________________ #
class Median_feature(com.Feature):
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
    ) -> None:
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
    ) -> Self:
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
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the moving median =======
        rolling_median = processed_series.rolling(window=window).apply(np.median, raw=False)

        # ======= III. Convert to pd.Series and Center =======
        rolling_median = (pd.Series(rolling_median, index=processed_series.index) / (processed_series + 1e-8)) - 1
        
        # ======= IV. Change Name =======
        rolling_median.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_median.index = data.index

        return rolling_median

#*____________________________________________________________________________________ #
class Minimum_feature(com.Feature):
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
    ) -> None:
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
    ) -> Self:
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
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the moving minimum =======
        rolling_min = processed_series.rolling(window=window).apply(np.min, raw=False)

        # ======= II. Convert to pd.Series and Center =======
        rolling_min = (pd.Series(rolling_min, index=processed_series.index) / (processed_series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_min.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_min.index = data.index

        return rolling_min
    
#*____________________________________________________________________________________ #
class Maximum_feature(com.Feature):
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
    ) -> None:
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
    ) -> Self:
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
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the moving maximum =======
        rolling_max = processed_series.rolling(window=window).apply(np.max, raw=False)

        # ======= II. Convert to pd.Series and Center =======
        rolling_max = (pd.Series(rolling_max, index=processed_series.index) / (processed_series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_max.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_max.index = data.index

        return rolling_max



#! ==================================================================================== #
#! ================================ Dispersion Features =============================== #
class Volatility_feature(com.Feature):
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
    ) -> None:
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
    ) -> Self:
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
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the moving volatility =======
        returns_series = processed_series.pct_change()
        rolling_vol = returns_series.rolling(window=window).apply(np.std, raw=False)

        # ======= II. Convert to pd.Series and Center =======
        rolling_vol = pd.Series(rolling_vol, index=processed_series.index)
        
        # ======= III. Change Name =======
        rolling_vol.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_vol.index = data.index

        return rolling_vol

#*____________________________________________________________________________________ #
class Skewness_feature(com.Feature):
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
    ) -> None:
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
    ) -> Self:
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
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the moving skewness =======
        returns_series = processed_series.pct_change()
        rolling_skew = returns_series.rolling(window=window).apply(lambda x: x.skew())

        # ======= II. Convert to pd.Series and Center =======
        rolling_skew = pd.Series(rolling_skew, index=processed_series.index)
        
        # ======= III. Change Name =======
        rolling_skew.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_skew.index = data.index

        return rolling_skew

#*____________________________________________________________________________________ #
class Kurtosis_feature(com.Feature):
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
    ) -> None:
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
    ) -> Self:
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
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the moving kurtosis =======
        returns_series = processed_series.pct_change()
        rolling_kurt = returns_series.rolling(window=window).apply(lambda x: x.kurtosis())

        # ======= II. Convert to pd.Series and Center =======
        rolling_kurt = pd.Series(rolling_kurt, index=processed_series.index) - 3 # Centering to excess kurtosis
        
        # ======= III. Change Name =======
        rolling_kurt.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_kurt.index = data.index

        return rolling_kurt

#*____________________________________________________________________________________ #
class Quantile_feature(com.Feature):
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
    ) -> None:
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
    ) -> Self:
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
        quantile: float,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ) -> pd.Series:
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
        
        processed_series = self.process_data(data=smoothed_series)

        # ======= II. Compute the moving quantile =======
        returns_series = processed_series.pct_change()
        rolling_quantile = returns_series.rolling(window=window).apply(lambda x: np.quantile(x, quantile))

        # ======= II. Convert to pd.Series and Center =======
        rolling_quantile = pd.Series(rolling_quantile, index=processed_series.index)
        
        # ======= III. Change Name =======
        rolling_quantile.name = f"{self.name}_{quantile}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_quantile.index = data.index

        return rolling_quantile

