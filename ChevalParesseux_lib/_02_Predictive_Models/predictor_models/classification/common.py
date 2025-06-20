from .... import utils

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Self, Optional
from joblib import Parallel, delayed



#! ==================================================================================== #
#! =================================== Base Models ==================================== #
class Classification_Model(ABC):
    """
    This class defines the core structure and interface for classification models. It is meant to be subclassed
    by specific model implementations.
    
    Subclasses must implement the following abstract methods:
        - __init__: Initializes the model with number of jobs.
        - set_params: Defines the parameters.
        - process_data: Applies preprocessing to the data.
        - fit: Fits the model to the training data.
        - predict: Makes predictions on the test data.
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the Model object.

        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        # ======= I. Initialize Class =======
        self.n_jobs = n_jobs

        # ======= II. Initialize Auxilaries =======
        self.params = {}
        self.metrics = {}
    
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
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series,
        **kwargs
    ) -> Self:
        """
        Fit the model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame | pd.Series): The input features for training.
            - y_train (pd.Series): The target variable for training.
            - **kwargs: Additional parameters for fitting the model.
        
        Returns:
            - None
        """
        ...
    
    #?_________________________________ Callable methods _________________________________ #
    @abstractmethod
    def predict(
        self,
        X_test: Union[pd.DataFrame, pd.Series],
        **kwargs
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Makes predictions on the test data.
        
        Parameters:
            - X_test (pd.DataFrame | pd.Series): The input features for testing.
            - **kwargs: Additional parameters for making predictions.
        
        Returns:
            - pd.DataFrame or pd.Series: The predicted values.
        """
        ...

    #?__________________________________ Common methods __________________________________ #
    