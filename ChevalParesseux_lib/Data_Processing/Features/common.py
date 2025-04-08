import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class Feature(ABC):
    """
    Abstract base class for all features.
    
    This class defines the core structure and interface for feature extraction. It is meant to be subclassed
    by specific feature implementations. Subclasses must implement the following abstract methods:
    
        - __init__: Initializes the feature with data, name, and optionally number of jobs.
        - set_params: Defines the parameter grid as a dictionary of lists.
        - process_data: Applies transformations or preprocessing to the data.
        - get_feature: Extracts the actual feature(s), returning a DataFrame.

    Main usage involves two core methods:
    
        - fit: Extracts all features from the parameter grid and filters them based on correlation.
        - extract: Returns extracted features, either filtered (if fitted) or raw (from parameter grid).
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
        
        Parameters:
            - data (pd.Series | pd.DataFrame | tuple): The raw input data.
            - name (str): The name identifier for the feature.
            - n_jobs (int): Number of parallel jobs to use during feature computation.
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
        """
        Sets the parameter grid for the feature extraction.
        
        Each parameter should be defined as a list of values, allowing grid search. 
        The parameters should be stored as a dictionary of lists.
        """
        pass

    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Preprocesses or filters the data before feature extraction.
        
        This may include operations like smoothing, cleaning, or filling missing values.
        Should return processed data in the same structure as input.
        """
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_feature(self):
        """
        Core method for feature extraction.
        
        Applies the actual computation using the parameters defined.
        Should return a DataFrame with feature values and proper column names.
        """
        pass
    
    #?____________________________________________________________________________________ #
    def fit(
        self,
        max_correlation: float = 0.95,
    ):
        """
        Extracts all features from the parameter grid and applies post-filtering.
        
        Steps:
            1. Extracts features using get_feature for every param combination.
            2. Applies a correlation filter to remove highly correlated features.
            3. Stores the filtered features.

        Parameters:
            - max_correlation (float): Max allowed correlation between two features.
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
        Returns the extracted features.

        If the feature has already been fitted, it returns the filtered version.
        Otherwise, computes and returns all features from the parameter grid.
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
    """
    Generates all possible combinations of parameters from a grid.
    
    This function takes a dictionary of parameters, where each key maps to a list of possible values,
    and returns a list of dictionaries representing the Cartesian product of all parameter combinations.

    Parameters:
        - params_grid (dict): Dictionary of parameters with lists of possible values.

    Returns:
        - params_list (list of dict): List of all parameter combinations.
    """
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
    Filters out highly correlated features based on a correlation threshold.

    For each pair of features, if their absolute correlation exceeds the max_correlation,
    one of them is removed to reduce redundancy and multicollinearity.

    Parameters:
        - features_df (pd.DataFrame): DataFrame containing feature columns.
        - max_correlation (float): Maximum allowed correlation between any two features.

    Returns:
        - filtered_features (pd.DataFrame): DataFrame with correlated features removed.
    """
    corr_matrix = features_df.corr().abs()

    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > max_correlation:
                to_drop.add(corr_matrix.columns[j])  
    
    filtered_features = features_df.drop(columns=to_drop).copy()

    return filtered_features
