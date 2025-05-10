import numpy as np
import pandas as pd


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
            - processed_data (pd.Series): The signs series as a message for entropy calculation.
        """
        processed_data = ent.get_movements_signs(series=data)
        
        return processed_data
    
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
        rolling_entropy = processed_series.rolling(window=window).apply(lambda x: ent.calculate_sample_entropy(series=x, sub_vector_size=sub_vector_size, threshold_distance=threshold_distance), raw=False)

        # ======= III. Convert to pd.Series and Center =======
        rolling_entropy = pd.Series(rolling_entropy, index=processed_series.index)
        
        # ======= IV. Change Name =======
        rolling_entropy.name = f"sample_entropy_{sub_vector_size}_{threshold_distance}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"
        
        return rolling_entropy

