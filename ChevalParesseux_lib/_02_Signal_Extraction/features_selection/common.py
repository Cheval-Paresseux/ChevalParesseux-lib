import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from abc import ABC, abstractmethod
from typing import Self, Union



#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class FeaturesSelector(ABC):
    """
    Base class for feature selection methods.
    
    This class provides an interface for feature selection methods, allowing for the extraction
    of features from data. It includes methods for setting parameters, processing data, fitting
    the model, and extracting features.
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1
    ):
        """
        Constructor for the FeaturesSelector class.
        
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
        Sets the parameter grid for the feature extraction.

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
    ) -> Union[tuple, pd.Series, pd.DataFrame]:
        """
        Processes the input data.
        
        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to be processed.
        
        Returns:
            - Union[tuple, pd.Series, pd.DataFrame]: A DataFrame containing the processed data.
        """
        ...
        
    #?__________________________________ User methods ____________________________________ #
    @abstractmethod
    def fit(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
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
    ) -> pd.DataFrame:
        """
        Extracts features from the given data.
        
        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data from which features are to be extracted.
        
        Returns:
            - pd.DataFrame: A DataFrame containing the extracted features.
        """
        ...
    
    