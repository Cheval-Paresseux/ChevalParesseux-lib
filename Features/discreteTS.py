import sys
sys.path.append("../")
import Measures as aux
from Features import common as cm

import pandas as pd
import numpy as np

#! ==================================================================================== #
#! ============================= Evolution Measure Features =========================== #
class D_average_feature(cm.Feature):
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
                "window_sizes": [5, 10, 30, 60, 120, 240],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self):
        self.processed_data = self.data.copy()

        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        discrete_series: pd.Series,
        window: int,
    ):
        # ======= I. Compute the rolling average =======
        rolling_avg = discrete_series.rolling(window=window + 1).mean()

        # ======= II. Convert to pd.Series and Normalize =======
        rolling_avg = pd.Series(rolling_avg, index=discrete_series.index)
        
        # ======= III. Change Name =======
        rolling_avg.name = f"average_{window}"

        return rolling_avg

#*____________________________________________________________________________________ #
class D_volatility_feature(cm.Feature):
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
                "window_sizes": [5, 10, 30, 60, 120, 240],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self):
        self.processed_data = self.data.copy()

        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        discrete_series: pd.Series,
        window: int,
    ):
        # ======= I. Compute the rolling volatility =======
        rolling_volatility = discrete_series.rolling(window=window + 1).std()

        # ======= II. Convert to pd.Series and Normalize =======
        rolling_volatility = pd.Series(rolling_volatility, index=discrete_series.index)
        
        # ======= III. Change Name =======
        rolling_volatility.name = f"volatility_{window}"

        return rolling_volatility

#*____________________________________________________________________________________ #
class D_changes_feature(cm.Feature):
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
                "window_sizes": [5, 10, 30, 60, 120, 240],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self):
        self.processed_data = self.data.copy()

        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        discrete_series: pd.Series,
        window: int,
    ):
        # ======= 0. Auxiliary function =======
        def compute_changes(series: pd.Series):
            diff_series = series.diff() ** 2
            changes_count = diff_series[diff_series > 0].count()

            return changes_count

        # ======= I. Compute the rolling changes =======
        rolling_changes = discrete_series.rolling(window=window + 1).apply(compute_changes, raw=False)

        # ======= II. Convert to pd.Series and Normalize =======
        rolling_changes = pd.Series(rolling_changes, index=discrete_series.index)
        
        # ======= III. Change Name =======
        rolling_changes.name = f"changes_{window}"

        return rolling_changes

#*____________________________________________________________________________________ #
class D_entropy_feature(cm.Feature):
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
                "window_sizes": [5, 10, 30, 60, 120, 240],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
            params=params,
            n_jobs=n_jobs,
            )

    #?____________________________________________________________________________________ #
    def process_data(self):
        self.processed_data = self.data.copy()

        return self.processed_data
    
    #?____________________________________________________________________________________ #
    def get_feature(
        discrete_series: pd.Series,
        window: int,
    ):
        # ======= I. Compute the rolling entropy features =======
        rolling_shannon = discrete_series.rolling(window=window + 1).apply(aux.get_shannon_entropy, raw=False)
        rolling_plugin = discrete_series.rolling(window=window + 1).apply(aux.get_plugin_entropy, raw=False)
        rolling_lempel_ziv = discrete_series.rolling(window=window + 1).apply(aux.get_lempel_ziv_entropy, raw=False)
        rolling_kontoyiannis = discrete_series.rolling(window=window + 1).apply(aux.get_kontoyiannis_entropy, raw=False)

        # ======= II. Convert to pd.Series =======
        rolling_shannon = pd.Series(rolling_shannon, index=discrete_series.index)
        rolling_plugin = pd.Series(rolling_plugin, index=discrete_series.index)
        rolling_lempel_ziv = pd.Series(rolling_lempel_ziv, index=discrete_series.index)
        rolling_kontoyiannis = pd.Series(rolling_kontoyiannis, index=discrete_series.index)
        
        # ======= III. Change Names =======
        rolling_shannon.name = f"shannon_entropy_{window}"
        rolling_plugin.name = f"plugin_entropy_{window}"
        rolling_lempel_ziv.name = f"lempel_ziv_entropy_{window}"
        rolling_kontoyiannis.name = f"kontoyiannis_entropy_{window}"

        return rolling_shannon, rolling_plugin, rolling_lempel_ziv, rolling_kontoyiannis

#*____________________________________________________________________________________ #