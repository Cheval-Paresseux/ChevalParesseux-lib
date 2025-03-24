import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class Labeller(ABC):
    @abstractmethod
    def __init__(
        self, 
        data: pd.Series, 
        params: dict, 
        n_jobs: int = 1
    ):
        # ======= I. Initialize Class =======
        self.data = data
        self.params = params
        self.n_jobs = n_jobs

        # ======= II. Initialize Auxilaries =======
        self.processed_data = None
        self.labels_series = None
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self):
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_labels(self):
        pass
    
    #?____________________________________________________________________________________ #
    def fit(self):
        pass
    
    #?____________________________________________________________________________________ #
    def extract(self):
        if self.labels_series:
            return self.labels_series
        else:
            params_grid = extract_universe(self.params)
            labels = Parallel(n_jobs=self.n_jobs)(delayed(self.get_labels)(self.processed_data, params) for params in params_grid)
            labels_df = pd.concat(labels, axis=1)

            return labels_df



#! ==================================================================================== #
#! ================================= Helper Functions ================================= #
def extract_universe(params_grid: dict): 
    # ======= 0. Define recursive function to generate all combinations =======
    def recursive_combine(keys, values, index, current_combination, params_list):
        if index == len(keys):
            # Base case: all parameters have been assigned a value
            params_list.append(current_combination.copy())
            return

        key = keys[index]
        for value in values[index]:
            current_combination[key] = value
            recursive_combine(keys, values, index + 1, current_combination, params_list)

    # ======= I. Initialize variables =======
    keys = list(params_grid.keys())
    values = list(params_grid.values())
    params_list = []

    # ======= II. Generate all combinations =======
    recursive_combine(keys, values, 0, {}, params_list)

    return params_list

#*____________________________________________________________________________________ #
def trend_filter(label_series: pd.Series, window: int):
    # ======= I. Create an auxiliary DataFrame =======
    auxiliary_df = pd.DataFrame()
    auxiliary_df["label"] = label_series
    
    # ======= II. Create a group for each label and extract size =======
    auxiliary_df["group"] = (auxiliary_df["label"] != auxiliary_df["label"].shift()).cumsum()
    group_sizes = auxiliary_df.groupby("group")["label"].transform("size")

    # ======= III. Filter the labels based on the group size =======
    auxiliary_df["label"] = auxiliary_df.apply(lambda row: row["label"] if group_sizes[row.name] >= window else 0, axis=1)
    labels_series = auxiliary_df["label"]
    
    return labels_series