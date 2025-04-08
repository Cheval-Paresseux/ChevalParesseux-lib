from ..Labelling import common as com

import numpy as np
import pandas as pd

#! ==================================================================================== #
#! =============================== BINARY LABELLERS ================================== #
class binaryMeta_labeller(com.Labeller):
    def __init__(
        self, 
        series: pd.DataFrame, 
        n_jobs: int = 1
    ):
        super().__init__(
            series=series, 
            n_jobs=n_jobs,
        )
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        trade_lock: bool = [True, False],
        noZero: bool = [True, False],
    ):
        self.params = {
            "trade_lock": trade_lock,
            "noZero": noZero,
        }
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(self):
        processed_data = self.series
        self.processed_data = processed_data
        
        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        trade_lock: bool = False,
        noZero: bool = False,
    ):
        meta_df = self.processed_data
        if trade_lock and not noZero:
            meta_df["meta_label"] = np.where(meta_df["predictions"] == 0, 1, np.where(meta_df["predictions"] == meta_df["label"], 1, 0))
            label_series = meta_df["meta_label"]
        elif noZero: 
            meta_df["meta_label"] = np.where((meta_df["predictions"] == meta_df["label"]) & (meta_df["predictions"] != 0), 1, 0)
            label_series = meta_df["meta_label"]
        else:
            meta_df["meta_label"] = np.where(meta_df["predictions"] == meta_df["label"], 1, 0)
            label_series = meta_df["meta_label"]
        
        return label_series
        
#*____________________________________________________________________________________ #

#! ==================================================================================== #
#! =============================== TRINARY LABELLERS =================================== #
class trinaryMeta_labeller(com.Labeller):
    def __init__(
        self, 
        series: pd.DataFrame, 
        n_jobs: int = 1
    ):
        super().__init__(
            series=series, 
            n_jobs=n_jobs,
        )
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        extension: bool = [True, False],
        noZero: bool = [True, False],
    ):
        """ Set parameters for the labeller.
        
        Parameters:
            extension (bool): Allows neutral (0,0) to be considered good if False.
            noZero (bool): Excludes cases where prediction = 0 from being "good".
        """
        self.params = {
            "extension": extension,
            "noZero": noZero,
        }
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(self):
        processed_data = self.series
        self.processed_data = processed_data
        
        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(self, extension: bool = False, noZero: bool = False):
        """ Generate labels based on different scoring rules.
        
        - extension: Allows neutral (0,0) to be considered good if False.
        - noZero: Excludes cases where prediction = 0 from being "good".
        
        Returns:
            pd.Series with labeled classifications.
        """
        # ======= 0. Get the data =======
        meta_df = self.processed_data

        # ======= I. Defining Conditions =======
        is_good = (meta_df["predictions"] == meta_df["label"])
        is_neutral = (meta_df["predictions"] == 0) & (meta_df["label"] != 0)
        is_ugly = ((meta_df["predictions"] == -1) & (meta_df["label"] == 1)) | ((meta_df["predictions"] == 1) & (meta_df["label"] == -1))
        
        # ======= II. Adjustments =======
        if noZero:
            is_good &= (meta_df["predictions"] != 0)

        if extension:
            is_good |= (meta_df["predictions"] == 0) & (meta_df["label"] == 0)

        # ======= III. Labelling =======
        meta_df["meta_label"] = np.select([is_good, is_neutral, is_ugly], [1, 0, -1], default=0)
        labels_series = meta_df["meta_label"]
        
        return labels_series
        
#*____________________________________________________________________________________ #

