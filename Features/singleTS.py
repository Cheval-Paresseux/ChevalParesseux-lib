import sys
sys.path.append("../")
import Measures as aux
from Features import common as cm
from Models import LinearRegression as reg

import numpy as np
import pandas as pd
# import pywt

#! ==================================================================================== #
#! ======================= Unscaled Smoothed-like Series Features ===================== #
class average_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_average = series.rolling(window=window + 1).apply(lambda x: np.mean(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_average = (pd.Series(rolling_average, index=series.index) / (series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_average.name = f"average_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_average


#*____________________________________________________________________________________ #
class median_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_median = series.rolling(window=window + 1).apply(lambda x: np.median(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_median = (pd.Series(rolling_median, index=series.index) / (series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_median.name = f"median_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_median


#*____________________________________________________________________________________ #
class minimum_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_min = series.rolling(window=window + 1).apply(lambda x: np.min(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_min = (pd.Series(rolling_min, index=series.index) / (series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_min.name = f"min_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_min
    
    
#*____________________________________________________________________________________ #
class maximum_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
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
class volatility_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
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
class skewness_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
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
class kurtosis_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
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
class quantile_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "quantile": [0.01, 0.05, 0.25, 0.75, 0.95, 0.99],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        quantile: float,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
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
class momentum_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_momentum = series.rolling(window=window + 1).apply(lambda x: aux.get_momentum(x[:window]))
        
        # ======= II. Convert to pd.Series and Center =======
        rolling_momentum = pd.Series(rolling_momentum, index=series.index)
        
        # ======= III. Change Name =======
        rolling_momentum.name = f"momentum_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_momentum


#*____________________________________________________________________________________ #
class Z_momentum_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_Z_momentum = series.rolling(window=window + 1).apply(lambda x: aux.get_Z_momentum(x[:window]))
        
        # ======= II. Convert to pd.Series and Center =======
        rolling_Z_momentum = pd.Series(rolling_Z_momentum, index=series.index)
        
        # ======= III. Change Name =======
        rolling_Z_momentum.name = f"Z_momentum_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_Z_momentum


#*____________________________________________________________________________________ #
class linear_tempReg_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= 0. Intermediate functions =======
        def compute_slope(series):
            _, coefficients, _, _ = aux.get_simple_TempReg(series)
            slope = coefficients[0]
            
            return slope

        def compute_T_stats(series):
            _, _, statistics, _ = aux.get_simple_TempReg(series)
            T_stats = statistics['T_stats'][0]
            
            return T_stats
        
        def compute_Pvalue(series):
            _, _, statistics, _ = aux.get_simple_TempReg(series)
            P_value = statistics['P_values'][0]
            
            return P_value
        
        def compute_R_squared(series):
            _, _, statistics, _ = aux.get_simple_TempReg(series)
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
class nonlinear_tempReg_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= 0. Intermediate functions =======
        def compute_slope(series):
            _, coefficients, _, _ = aux.get_quad_TempReg(series)
            slope = coefficients[0]
            
            return slope
        
        def compute_acceleration(series):
            _, coefficients, _, _ = aux.get_quad_TempReg(series)
            acceleration = coefficients[1]
            
            return acceleration

        def compute_T_stats(series):
            _, _, statistics, _ = aux.get_quad_TempReg(series)
            T_stats = statistics['T_stats'][0]
            
            return T_stats
        
        def compute_Pvalue(series):
            _, _, statistics, _ = aux.get_quad_TempReg(series)
            P_value = statistics['P_values'][0]
            
            return P_value
        
        def compute_R_squared(series):
            _, _, statistics, _ = aux.get_quad_TempReg(series)
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
class hurst_exponent_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "power": [3, 4, 5, 6, 7, 8],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        power: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
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
            statistics, _ = model.get_statistics()
            tstat = statistics['T_stats'][0]
            pvalue = statistics['P_values'][0]
            
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
class entropy_feature(cm.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
            
        elif smoothing_method == "ewma":
            processed_data = aux.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
            
        elif smoothing_method == "average":
            processed_data = aux.average_smoothing(price_series=self.data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        signs_series = aux.get_movements_signs(series=series)

        # ======= II. Compute the rolling entropy features =======
        rolling_shannon = signs_series.rolling(window=window + 1).apply(aux.get_shannon_entropy, raw=False)
        rolling_plugin = signs_series.rolling(window=window + 1).apply(aux.get_plugin_entropy, raw=False)
        rolling_lempel_ziv = signs_series.rolling(window=window + 1).apply(aux.get_lempel_ziv_entropy, raw=False)
        rolling_kontoyiannis = signs_series.rolling(window=window + 1).apply(aux.get_kontoyiannis_entropy, raw=False)

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


#*____________________________________________________________________________________ #
# def wavelets_features(
#     price_series: pd.Series,
#     wavelet_window: int,
#     wav_family: list = [],
#     decomposition_level: int = 2,
# ):
#     # ======= 0. Initialize the input series as a dataframe to store the wavelets =======
#     if len(wav_family) == 0:
#         wav_family = ["haar", "db1", "db2", "db3", "db4", "sym2", "sym3", "sym4", "sym5", "coif1", "coif2", "coif3", "coif4", "bior1.1", "bior1.3", "bior1.5", "bior2.2", "rbio1.1", "rbio1.3", "rbio1.5"]

#     if price_series.name is None:
#         price_series.name = "close"
#     price_df = price_series.to_frame().copy()
#     price_df.rename(columns={price_series.name: "close"}, inplace=True)

#     # ======= I. Compute the wavelets for each family =======
#     for wavelet in wav_family:
#         # I.1 Initialize the lists to store the wavelet features
#         mean = [[None] * wavelet_window for _ in range(decomposition_level)]
#         median = [[None] * wavelet_window for _ in range(decomposition_level)]
#         std = [[None] * wavelet_window for _ in range(decomposition_level)]
#         max = [[None] * wavelet_window for _ in range(decomposition_level)]
#         min = [[None] * wavelet_window for _ in range(decomposition_level)]
#         # => Each inside list corresponds to a decomposition level, and each element of those inside lists corresponds to a window

#         # I.2 Compute the wavelet features
#         for index in range(wavelet_window, price_df.shape[0]):
#             # I.2.i Extract the rolling window of the price series and compute the wavelet coefficients
#             price_window = price_df.iloc[index - wavelet_window : index, 0].copy()
#             coeffs = pywt.wavedec(
#                 price_window,
#                 wavelet,
#                 level=decomposition_level,
#             )

#             # I.2.ii Compute the wavelet features for each decomposition level
#             for level in range(decomposition_level):
#                 # We start at level 1 because the first element of the coeffs list is the approximation coefficients
#                 mean[level].append(np.mean(coeffs[level + 1]))
#                 median[level].append(np.median(coeffs[level + 1]))
#                 std[level].append(np.std(coeffs[level + 1]))
#                 max[level].append(np.max(coeffs[level + 1]))
#                 min[level].append(np.min(coeffs[level + 1]))

#         # I.3 Store the wavelet features in the dataframe
#         for level in range(decomposition_level):
#             price_df.loc[:, f"{wavelet}_{level + 1}_mean"] = mean[level]
#             price_df.loc[:, f"{wavelet}_{level + 1}_median"] = median[level]
#             price_df.loc[:, f"{wavelet}_{level + 1}_std"] = std[level]
#             price_df.loc[:, f"{wavelet}_{level + 1}_max"] = max[level]
#             price_df.loc[:, f"{wavelet}_{level + 1}_min"] = min[level]

#     #  ======= II. Convert to Series and Center =======
#     price_df.drop(labels="close", axis=1, inplace=True)
#     features_columns = price_df.columns

#     for feature in features_columns:
#         price_df[feature] = price_df[feature] - price_df[feature].rolling(window=252).mean()

#     features_tuple = tuple([price_df[feature] for feature in features_columns])

#     return features_tuple

