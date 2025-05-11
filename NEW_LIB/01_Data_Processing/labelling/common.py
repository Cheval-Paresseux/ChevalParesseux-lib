from ...utils import calculations as calc

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Self
from joblib import Parallel, delayed



#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class Labeller(ABC):
    """
    This class defines the core structure and interface for labels extraction. It is meant to be subclassed
    by specific labeller implementations. 
    Subclasses must implement the following abstract methods:
        - __init__: Initializes the feature with number of jobs.
        - set_params: Defines the parameter grid as a dictionary of lists.
        - process_data: Applies preprocessing to the data.
        - get_labels: Extracts the actual labels, returning a DataFrame.

    Main usage involves one core methods:
        - smooth_data: Applies optional smoothing to the input data before feature computation.
        - extract: Returns extracted labels.
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        name: str,
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the Labeller object.

        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        # ======= I. Initialize Class =======
        self.name = name
        self.n_jobs = n_jobs

        # ======= II. Initialize Auxilaries =======
        self.params = {}
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def set_params(
        self,
        **kwargs
    ) -> Self:
        """
        Sets the parameter grid for the labels extraction.

        Parameters:
            - **kwargs: Each parameter should be a list of possible values.
                    Example: labeller.set_params(window=[5, 10], threshold=[3, 4])

        Returns:
            - Self: The instance of the class with the parameter grid set.
        """
        ...

    #?________________________________ Auxiliary methods _________________________________ #
    @abstractmethod
    def process_data(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
        **kwargs
    ) -> Union[tuple, pd.DataFrame, pd.Series]:
        """
        Preprocesses the data before labels extraction.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to be processed.
            - **kwargs: Additional parameters for the data processing.

        Returns:
            - tuple or pd.DataFrame or pd.Series: The processed data ready for feature extraction.
        """
        ...
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_labels(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
        **kwargs
    ) -> pd.Series:
        """
        Core method for labels extraction.
        
        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to extract the labels from
            - **kwargs: Additional parameters for the labels extraction.
        
        Returns:
            - pd.Series : The extracted labels as a pd.Series.
        """
        ...
        
    #?_________________________________ Callable methods _________________________________ #
    def smooth_data(
        self, 
        data: pd.Series,
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ) -> pd.Series:
        """
        Applies optional smoothing to the input data before labels computation.

        Parameters:
            - data (pd.Series): The input data to be processed.
            - smoothing_method (str): Type of smoothing to apply. Options: "ewma", "average", or None.
            - window_smooth (int): Size of the smoothing window.
            - lambda_smooth (float): EWMA decay parameter in [0, 1].

        Returns:
            - smoothed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        """
        # ======= I. Check if any smoothing should be applied =======
        if smoothing_method is None:
            return data
        
        # ======= II. Compute the smoothed series =======
        elif smoothing_method == "ewma":
            smoothed_data = calc.ewma_smoothing(price_series=data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            smoothed_data = calc.average_smoothing(price_series=data, window=window_smooth)
            
        else:
            raise ValueError("Smoothing method not recognized")
        
        return smoothed_data
    
    #?____________________________________________________________________________________ #
    def extract(
        self, 
        data: Union[tuple, pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Main method to extract labels.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to extract the labels from
        
        Returns:
            - labels_df (pd.DataFrame): The extracted labels as a DataFrame.
        """
        # ======= I. Extract the Parameters Universe =======
        params_grid = calc.get_dict_universe(self.params)

        # ======= II. Extract the labels for each Parameters =======
        labels = Parallel(n_jobs=self.n_jobs)(delayed(self.get_labels)(data, **params) for params in params_grid)

        # ======= III. Create a DataFrame with the labels =======
        labels_df = pd.concat(labels, axis=1)

        return labels_df



#! ==================================================================================== #
#! =============================== Helper Functions =================================== #
def segment_length_filter(
    label_series: pd.Series, 
    window: int
) -> pd.Series:
    """
    Filters out short-lived label segments that are smaller than a given window.

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
    auxiliary_df["label"] = auxiliary_df.apply(lambda row: row["label"] if group_sizes.get(row.name, 0) >= window else 0, axis=1)
    labels_series = auxiliary_df["label"]
    
    return labels_series
