import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from abc import ABC, abstractmethod
from typing import Self, Union



#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class PredictorTuning(ABC):
    """
    Base class for Tuning predictors models.

    This class provides a template for tuning machine learning models using variations of grid search.
    It allows for parallel processing and can be extended to implement specific tuning strategies.
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1
    ):
        """
        Constructor for the PredictorTuning class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use during computation.
        """
        # ======= I. Initialize Class =======
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
        Sets the parameters for the extraction.

        Parameters:
            - **kwargs: additional parameters.

        Returns:
            - Self: The instance of the class with the parameters set.
        """
        ...
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
    ) -> Union[pd.DataFrame, list]:
        """
        Processes the input data and returns a DataFrame.
        
        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to be processed.
        
        Returns:
            - Union[pd.DataFrame, list]: A DataFrame or list containing the processed data.
        """
        ...
    
    #?__________________________________ User methods ____________________________________ #
    @abstractmethod
    def fit(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame, list],
    ) -> Self:
        """
        Extracts selection rules from the given data.
        
        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data from which rules are to be extracted.
        
        Returns:
            - Self: The instance of the class with the rules set.
        """
        ...

    #?____________________________________________________________________________________ #
    @abstractmethod
    def extract(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
    ) -> Union[tuple, pd.Series, pd.DataFrame]:
        """
        Extracts features from the given data.
        
        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data from which features are to be extracted.
        
        Returns:
            - Union[tuple, pd.Series, pd.DataFrame]: The extracted features.
        """
        ...
    
    