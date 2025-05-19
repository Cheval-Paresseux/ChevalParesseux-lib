from ..features import common as com
from ... import utils

import numpy as np
import pandas as pd
from typing import Self



#! ==================================================================================== #
#! ============================= Signal Processing Features =========================== #
class Shannon_entropy_feature(com.Feature):
    """
    Rolling Shannon Entropy Feature Extraction

    This class computes the Shannon entropy over a rolling window of a time series.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving entropies feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "shannon_entropy" , 
        n_jobs: int = 1
    ) -> None:
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
    ) -> Self:
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
    ) -> pd.Series:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The signs series as a message for entropy calculation.
        """
        signs_series = utils.get_movements_signs(series=data)
        
        return signs_series
    
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
        Computes rolling Shannon entropy feature from the series.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - window (int): Rolling window size for entropy calculation.
            - smoothing_method (str): Smoothing method to apply prior to entropy calculation.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - rolling_shannon (pd.Series): Series of rolling Shannon entropy values.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series)
        
        # ======= II. Compute the rolling entropy feature =======
        rolling_shannon = processed_series.rolling(window=window).apply(utils.get_shannon_entropy, raw=False)

        # ======= III. Convert to pd.Series =======
        rolling_shannon = pd.Series(rolling_shannon, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_shannon.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_shannon.index = data.index

        return rolling_shannon

#*____________________________________________________________________________________ #
class Plugin_entropy_feature(com.Feature):
    """
    Rolling Plugin Entropy Feature Extraction

    This class computes the Plugin entropy over a rolling window of a time series.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving entropies feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "plugin_entropy" , 
        n_jobs: int = 1
    ) -> None:
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
    ) -> Self:
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
    ) -> pd.Series:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The signs series as a message for entropy calculation.
        """
        signs_series = utils.get_movements_signs(series=data)
        
        return signs_series
    
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
        Computes rolling Plugin entropy feature from the series.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - window (int): Rolling window size for entropy calculation.
            - smoothing_method (str): Smoothing method to apply prior to entropy calculation.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - rolling_plugin (pd.Series): Series of rolling Plugin entropy values.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series)
        
        # ======= II. Compute the rolling entropy feature =======
        rolling_plugin = processed_series.rolling(window=window).apply(utils.get_plugin_entropy, raw=False)

        # ======= III. Convert to pd.Series =======
        rolling_plugin = pd.Series(rolling_plugin, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_plugin.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_plugin.index = data.index

        return rolling_plugin

#*____________________________________________________________________________________ #
class LempelZiv_entropy_feature(com.Feature):
    """
    Rolling Lempel-Ziv Entropy Feature Extraction

    This class computes the Lempel-Ziv entropy over a rolling window of a time series.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving entropies feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "lempelZiv_entropy" , 
        n_jobs: int = 1
    ) -> None:
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
    ) -> Self:
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
    ) -> pd.Series:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The signs series as a message for entropy calculation.
        """
        signs_series = utils.get_movements_signs(series=data)
        
        return signs_series
    
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
        Computes rolling Lempel-Ziv entropy feature from the series.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - window (int): Rolling window size for entropy calculation.
            - smoothing_method (str): Smoothing method to apply prior to entropy calculation.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - rolling_lempelZiv (pd.Series): Series of rolling Lempel-Ziv entropy values.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series)
        
        # ======= II. Compute the rolling entropy feature =======
        rolling_lempelZiv = processed_series.rolling(window=window).apply(utils.get_lempel_ziv_entropy, raw=False)

        # ======= III. Convert to pd.Series =======
        rolling_lempelZiv = pd.Series(rolling_lempelZiv, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_lempelZiv.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_lempelZiv.index = data.index

        return rolling_lempelZiv

#*____________________________________________________________________________________ #
class Kontoyiannis_entropy_feature(com.Feature):
    """
    Rolling Kontoyiannis Entropy Feature Extraction

    This class computes the Kontoyiannis entropy over a rolling window of a time series.
    It inherits from the Feature base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_feature : compute the moving entropies feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "kontoyiannis_entropy" , 
        n_jobs: int = 1
    ) -> None:
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
    ) -> Self:
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
    ) -> pd.Series:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The signs series as a message for entropy calculation.
        """
        signs_series = utils.get_movements_signs(series=data)
        
        return signs_series
    
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
        Computes rolling Kontoyiannis entropy feature from the series.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - window (int): Rolling window size for entropy calculation.
            - smoothing_method (str): Smoothing method to apply prior to entropy calculation.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Decay factor for EWMA.

        Returns:
            - rolling_kontoyiannis (pd.Series): Series of rolling Kontoyiannis entropy values.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series)
        
        # ======= II. Compute the rolling entropy feature =======
        rolling_kontoyiannis = processed_series.rolling(window=window).apply(utils.get_kontoyiannis_entropy, raw=False)

        # ======= III. Convert to pd.Series =======
        rolling_kontoyiannis = pd.Series(rolling_kontoyiannis, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_kontoyiannis.name = f"{self.name}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_kontoyiannis.index = data.index

        return rolling_kontoyiannis

#*____________________________________________________________________________________ #
class Sample_entropy_feature(com.Feature):
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
    ) -> None:
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
    ) -> Self:
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
    ) -> pd.Series:
        """
        Applies preprocessing to the input data before feature extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The signs series as a message for entropy calculation.
        """
        signs_series = utils.get_movements_signs(series=data)
        
        return signs_series
    
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
    ) -> pd.Series:
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
        
        processed_series = self.process_data(data=smoothed_series)
        
        # ======= II. Compute the rolling entropy features =======
        rolling_sample = processed_series.rolling(window=window).apply(lambda x: utils.get_sample_entropy(series=x, sub_vector_size=sub_vector_size, threshold_distance=threshold_distance), raw=False)

        # ======= III. Convert to pd.Series =======
        rolling_sample = pd.Series(rolling_sample, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_sample.name = f"{self.name}_{sub_vector_size}_{threshold_distance}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        rolling_sample.index = data.index

        return rolling_sample

