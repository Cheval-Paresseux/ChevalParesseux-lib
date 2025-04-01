import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class Labeller(ABC):
    """
    This class is the base class for all labellers.
    It should be inherited by all labellers and should implement the following methods:
        - process_data: This method should be used to process the data before extracting the labels.
        - get_labels: This method does the core computation for the labels extraction.
        
    After being initialized, the labels are extracted using two ways :
        - fit : #TODO 
        - extract : This method is the main one to be called to extract the labels. If the labels have been fitted before, it will return the filtered features. 
                    Otherwise it returns all the grid labels.
    """
    @abstractmethod
    def __init__(
        self, 
        data: pd.Series, 
        params: dict, 
        n_jobs: int = 1
    ):
        """
        Constructor for the Labeller class.
        For each labeller, the constructor should initialize the following attributes:
            
            - data (pd.Series): The series to be processed.
            - params (dict): The parameters for the labeller, if None, default parameters should be used.
            - n_jobs (int): The number of jobs to run in parallel.
        """
        # ======= I. Initialize Class =======
        self.data = data
        self.params = params
        self.n_jobs = n_jobs

        # ======= II. Initialize Auxilaries =======
        self.processed_data = None
        self.labels = None
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self):
        """
        This method is defined for each labeller to process the data before extracting the features.
        It usually serves to clean the data, fill missing values..., etc. apply filters to the data.
        
        It should be called inside the get_labels method to process the data before extracting the labels and return a pd.Series.
        """
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_labels(self):
        """
        This method is defined for each feature to make the core computation for the labels extraction.
        It should take each parameter individually and the data as input. The process_data method should be called to process the data before extracting the labels.
        
        It should output a pd.DataFrame with the labels as columns.
        """
        pass
    
    #?____________________________________________________________________________________ #
    def fit(self):
        """
        This method is used to extract the labels from the parameter grid and apply a filter.
        """
        # ======= I. Extract params_grid =========
        params_grid = extract_universe(self.params)
        labels = Parallel(n_jobs=self.n_jobs)(delayed(self.get_feature)(**params) for params in params_grid)
        labels_df = pd.concat(labels, axis=1)

        # ======= II. Apply the optimizing functions =========
        #TODO

        # ======= III. Save the features =======
        self.labels = labels_df

        return self.labels
    
    #?____________________________________________________________________________________ #
    def extract(self):
        """
        This method is the main one to be called to extract the labels.
        If the features have been fitted before, it will return the filtered features. Otherwise it returns all the grid features.
        """
        # ======= I. Check if the labels have been fitted =======
        if self.labels:
            return self.labels
        
        # ======= II. Extract labels from the parameter grid =======
        else:
            params_grid = extract_universe(self.params)
            labels = Parallel(n_jobs=self.n_jobs)(delayed(self.get_labels)(**params) for params in params_grid)
            labels_df = pd.concat([series.to_frame().rename(lambda col: f"set_{i}", axis=1) for i, series in enumerate(labels)], axis=1)
            
            self.labels = labels_df

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
    """
    This function applies a filter to the labels by removing the groups of labels that are smaller than the window size.
    Inputs:
        - label_series (pd.Series): The series of labels to be filtered.
        - window (int): The minimum size of the group to be kept.
    Outputs:
        - labels_series (pd.Series): The filtered series of labels.
    """
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
