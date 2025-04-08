import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from typing import Union

#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class Labeller(ABC):
    """
    Base Class for Label Extraction in Financial Time Series

    This abstract base class defines the interface for all labellers used to extract
    target variables or classification labels from time series data. It provides 
    structure for data processing, parameter configuration, and label extraction.

    Key Responsibilities:
        - process_data: Prepares the input time series (e.g., smoothing, cleaning).
        - get_labels: Performs core computation for extracting labels.
        - fit: [Optional] Fit a labeler using grid search and optimization (to be implemented).
        - extract: Main method for extracting labels using the configured parameter grid.

    Intended to be subclassed by specific labeling strategies (e.g., trend, volatility).
    """
    @abstractmethod
    def __init__(
        self, 
        series: Union[pd.Series, pd.DataFrame], 
        n_jobs: int = 1
    ):
        """
        Initializes the Labeller object.

        Parameters:
            - series (pd.Series or pd.DataFrame): The input time series to label.
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        # ======= I. Initialize Class =======
        self.series = series
        self.n_jobs = n_jobs

        # ======= II. Initialize Auxilaries =======
        self.params = None
        self.processed_data = None
        self.labels = None
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def set_params(self):
        """
        Configures the parameter grid used in label extraction.

        Should define self.params as a dictionary with all parameter values
        to be combined via cartesian product during parallel label extraction.
        """
        pass

    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self):
        """
        Prepares the raw time series for label extraction.

        This method can apply preprocessing such as cleaning missing values,
        filtering, or smoothing. It should return a processed pd.Series or
        pd.DataFrame and be called within get_labels.
        """
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_labels(self):
        """
        Computes labels using the processed data and a specific parameter set.

        This is the core labeling logic for each parameter combination.
        Should return a pd.Series or pd.DataFrame with one or more label columns.
        """
        pass
    
    #?____________________________________________________________________________________ #
    def fit(self):
        """
        Extracts labels using the parameter grid and applies filtering or optimization.

        This method performs parallel label extraction for all parameter combinations,
        then optionally applies model selection or filtering logic (to be implemented).

        Returns:
            - labels_df (pd.DataFrame): Optimized or filtered label set.
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
        Extracts all labels based on the defined parameter grid.

        If labels have already been computed (via `fit`), returns the stored result.
        Otherwise, extracts all label sets by running `get_labels` across the grid.

        Returns:
            - labels_df (pd.DataFrame): DataFrame containing one column per label set.
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
    """
    Generates the full parameter universe from a grid dictionary.

    This function takes a dictionary of parameter lists and returns a list of all
    possible parameter combinations, similar to a grid search. Each combination is 
    returned as a dictionary mapping parameter names to values.

    Parameters:
        - params_grid (dict): Dictionary of the form {"param1": [v1, v2], "param2": [w1, w2, w3], ...}

    Returns:
        - params_list (List[dict]): List of dictionaries, each representing one parameter combination.
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
def trend_filter(label_series: pd.Series, window: int):
    """
    Filters out short-lived label segments that are smaller than a given window.

    This function removes segments of consecutive identical labels whose length
    is smaller than the specified `window` size. These short segments are replaced
    with a neutral label (0).

    Parameters:
        - label_series (pd.Series): Series of integer or categorical labels.
        - window (int): Minimum segment length required to preserve a label.

    Returns:
        - labels_series (pd.Series): Filtered series with small segments replaced by 0.
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
