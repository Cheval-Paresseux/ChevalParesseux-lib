from ... import utils

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Self, Optional
from joblib import Parallel, delayed



#! ==================================================================================== #
#! =================================== Base Models ==================================== #
class SignalProcessor(ABC):
    """
    Base class for signal processing models.
    
    Subclasses should implement the following methods:
        - __init__: Initialize the model with parameters.
        - set_params: Set the parameters for the model.
        - process_data: Preprocess the data.
        - fit: Fit the model to the training data if necessary.
        - predict: Make predictions on the test data.
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the SignalProcessor object.

        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
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
        Sets the parameter for the model.

        Parameters:
            - **kwargs: Additional parameters to be set.

        Returns:
            - Self: The instance of the class with the parameter set.
        """
        ...

    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
        **kwargs
    ) -> Union[tuple, pd.DataFrame, pd.Series]:
        """
        Preprocesses the data.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to be processed.
            - **kwargs: Additional parameters for the data processing.

        Returns:
            - tuple or pd.DataFrame or pd.Series: The processed data ready for extraction.
        """
        ...

    #?________________________________ Auxiliary methods _________________________________ #
    @abstractmethod
    def fit(
        self,
        **kwargs
    ) -> Self:
        """
        Fit the model to the training data.
        
        Parameters:
            - **kwargs: Additional parameters for fitting the model.
        
        Returns:
            - None
        """
        ...
    
    #?_________________________________ Callable methods _________________________________ #
    @abstractmethod
    def predict(
        self,
        data: Union[pd.DataFrame, pd.Series],
        **kwargs
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Makes predictions on the test data.
        
        Parameters:
            - data (pd.DataFrame | pd.Series): The input data for signal processing.
            - **kwargs: Additional parameters for signal processing.
        
        Returns:
            - pd.DataFrame or pd.Series: The predicted values.
        """
        ...

