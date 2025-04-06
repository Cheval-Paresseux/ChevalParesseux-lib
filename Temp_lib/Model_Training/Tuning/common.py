import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from abc import ABC, abstractmethod

class GridSearch(ABC):
    @abstractmethod
    def __init__(self, model):
        self.model = model

    #?____________________________________________________________________________________ #
    @abstractmethod
    def set_params_grid(self):
        pass

    #?____________________________________________________________________________________ #
    @abstractmethod
    def set_params_sets(self):
        pass

    #?____________________________________________________________________________________ #
    @abstractmethod
    def transform_data(self):
        pass

    #?____________________________________________________________________________________ #
    @abstractmethod
    def fit(self):
        pass

    #?____________________________________________________________________________________ #
    def score(self):
        pass

    
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