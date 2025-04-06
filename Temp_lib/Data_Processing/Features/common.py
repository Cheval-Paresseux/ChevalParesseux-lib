import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class Feature(ABC):
    """
    This class is the base class for all features.
    It should be inherited by all features and should implement the following methods:
        - process_data: This method should be used to process the data before extracting the features.
        - get_feature: This method does the core computation for the feature extraction. It should output a pd.DataFrame with the feature name as columns.
        - init: This method should initialize the class with the data, name, params and n_jobs. It is recommended to put default values for the params and n_jobs arguments.
    
    After being initialized, the features are extracted using two ways :
        - fit : This method serves to extract every features from a parameter grid and apply a filter. #TODO: Improve how it fits the features.
        - extract : This method is the main one to be called to extract the features. If the features have been fitted before, it will return the filtered features. 
                    Otherwise it returns all the grid features.
    """
    @abstractmethod
    def __init__(
        self, 
        data: Union[tuple, pd.Series, pd.DataFrame], 
        name: str, 
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
        self.n_jobs = n_jobs

        # ======= II. Initialize Auxilaries =======
        self.params = None
        self.processed_data = None
        self.features = None
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def set_params(self):
        pass

    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        This method is defined for each feature to process the data before extracting the features.
        It usually serves to clean the data, fill missing values..., etc. apply filters to the data.
        
        It should be called inside the get_feature method to process the data before extracting the features and return a pd.DataFrame or pd.Series.
        """
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_feature(self):
        """
        This method is defined for each feature to make the core computation for the feature extraction.
        It should take each parameter individually and the data as input. The process_data method should be called to process the data before extracting the features.
        
        It should output a pd.DataFrame with the feature name as columns.
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
        # ======= I. Extract params_grid =========
        params_grid = extract_universe(self.params)
        features = Parallel(n_jobs=self.n_jobs)(delayed(self.get_feature)(**params) for params in params_grid)
        features_df = pd.concat(features, axis=1)

        # ======= II. Apply the optimizing functions =========
        # 1. Correlation filter => Eliminate the features that are too correlated
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
        # ======= I. Check if features have been fitted =======
        if self.features is not None:
            return self.features
        
        # ======= II. Extract features from the parameter grid =======
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
def correlation_filter(features_df: pd.DataFrame, max_correlation: float):
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
