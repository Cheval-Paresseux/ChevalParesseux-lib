from ...utils import function_tools as tools

import numpy as np
import pandas as pd
from typing import Union
from typing import Self
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class Feature(ABC):
    """
    Abstract base class for all features.
    
    This class defines the core structure and interface for feature extraction. It is meant to be subclassed
    by specific feature implementations. 
    Subclasses must implement the following abstract methods:
        - __init__: Initializes the feature with name, and optionally number of jobs.
        - set_params: Defines the parameter grid as a dictionary of lists.
        - process_data: Applies transformations or preprocessing to the data.
        - get_feature: Extracts the actual feature(s), returning a DataFrame.

    Main usage involves one core methods:
        - extract: Returns extracted features, either filtered (if fitted) or raw (from parameter grid).
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        name: str, 
        n_jobs: int = 1
    ):
        """
        Constructor for the Feature class.
        
        Parameters:
            - name (str): The name identifier for the feature.
            - n_jobs (int): Number of parallel jobs to use during feature computation.
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
        Sets the parameter grid for the feature extraction.

        Parameters:
            - **kwargs: Each parameter should be a list of possible values.
                    Example: feature.set_params(window=[5, 10], threshold=[3, 4])

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
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Preprocesses or filters the data before feature extraction.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to be processed.
            - **kwargs: Additional parameters for the data processing.

        Returns:
            - pd.DataFrame or pd.Series: The processed data ready for feature extraction.
        """
        ...
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_feature(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
        **kwargs
    ) -> pd.Series:
        """
        Core method for feature extraction.
        
        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to extract the feature from
            - **kwargs: Additional parameters for the feature extraction.
        
        Returns:
            - pd.Series : The extracted feature as a pd.Series.
        """
        ...
       
    #?_________________________________ Callable methods _________________________________ #
    def extract(
        self, 
        data: Union[tuple, pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Main method to extract features.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to extract the feature from
        
        Returns:
            - features_df (pd.DataFrame): The extracted features as a DataFrame.
        """
        # ======= I. Extract the Parameters Universe =======
        params_grid = tools.get_dict_universe(self.params)

        # ======= II. Extract the features for each Parameters =======
        features = Parallel(n_jobs=self.n_jobs)(delayed(self.get_feature)(data, **params) for params in params_grid)

        # ======= III. Create a DataFrame with the features =======
        features_df = pd.concat(features, axis=1)

        return features_df
