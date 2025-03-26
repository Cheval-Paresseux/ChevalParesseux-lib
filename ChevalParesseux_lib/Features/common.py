import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class Feature(ABC):
    @abstractmethod
    def __init__(
        self, 
        data: Union[tuple, pd.Series, pd.DataFrame], 
        name: str, 
        params: dict, 
        n_jobs: int = 1
    ):
        """
        Constructor for the Feature class.
        For each feature, the constructor should initialize the following attributes:
            
            - data (pd.Series | pd.DataFrame): The data to be processed.
            - name (str): The name of the feature.
            - params (dict): The parameters for the feature, if None, default parameters should be used.
            - n_jobs (int): The number of jobs to run in parallel.
        """
        # ======= I. Initialize Class =======
        self.data = data
        self.name = name
        self.params = params
        self.n_jobs = n_jobs

        # ======= II. Initialize Auxilaries =======
        self.processed_data = None
        self.features = None
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self):
        """
        This method should be used to process the data before extracting the features.
        The main purpose of this method is to ensure that the data is in the correct format to be passed into the get_feature method.
        """
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_feature(self):
        """
        This method does the core computation for the feature extraction.
        It should take the processed data and the parameters for the feature and return the feature series.
        """
        pass
    
    #?____________________________________________________________________________________ #
    def fit(
        self,
        max_correlation: float = 0.95,
    ):
        """
        This method serves to extract every features from a parameter grid and apply a filter.

            - max_correlation (float): The maximum correlation allowed between two features.
        """
        # ======= 0. Extract params_grid =========
        params_grid = extract_universe(self.params)

        # ======= I. Compute the different feature series =========
        features = Parallel(n_jobs=self.n_jobs)(delayed(self.get_feature)(**params) for params in params_grid)

        # ======= II. Eliminate the features that are too correlated =========
        features_df = pd.concat(features, axis=1)
        features_df = correlation_filter(features_df, max_correlation)

        # ======= III. Save the features =======
        self.features = features_df

        return self.features
    
    #?____________________________________________________________________________________ #
    def extract(self):
        """
        This method is the main one to be called to extract the features.
        If the features have been fitted before, it will return the filtered features. Otherwise it returns all the grid features.
        """
        if self.features is not None:
            return self.features
        else:
            params_grid = extract_universe(self.params)
            features = Parallel(n_jobs=self.n_jobs)(delayed(self.get_feature)(**params) for params in params_grid)
            features_df = pd.concat(features, axis=1)

            return features_df



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
def correlation_filter(features_df: pd.DataFrame, max_correlation: float = 0.95):
    """
    This function filters the features based on the correlation between them.

        - features_df (pd.DataFrame): The features to be filtered.
        - max_correlation (float): The maximum correlation allowed between two features.
    """
    corr_matrix = features_df.corr().abs()

    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > max_correlation:
                to_drop.add(corr_matrix.columns[j])  
    
    filtered_features = features_df.drop(columns=to_drop).copy()

    return filtered_features
