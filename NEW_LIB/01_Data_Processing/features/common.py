from ...utils import calculations as calc

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
        - process_data: Applies preprocessing to the data.
        - get_feature: Extracts the actual feature(s), returning a DataFrame.

    Main usage involves one core methods:
        - smooth_data: Applies optional smoothing to the input data before feature computation.
        - extract: Returns extracted features.
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
    ) -> Union[tuple, pd.DataFrame, pd.Series]:
        """
        Preprocesses the data before feature extraction.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to be processed.
            - **kwargs: Additional parameters for the data processing.

        Returns:
            - tuple or pd.DataFrame or pd.Series: The processed data ready for feature extraction.
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
       
    #?____________________________________________________________________________________ #
    def smooth_data(
        self, 
        data: pd.Series,
        smoothing_method: str = None, 
        window_smooth: int = None, 
        lambda_smooth: float = None
    ):
        """
        Applies optional smoothing to the input data before feature computation.

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
        params_grid = calc.get_dict_universe(self.params)

        # ======= II. Extract the features for each Parameters =======
        features = Parallel(n_jobs=self.n_jobs)(delayed(self.get_feature)(data, **params) for params in params_grid)

        # ======= III. Create a DataFrame with the features =======
        features_df = pd.concat(features, axis=1)

        return features_df
