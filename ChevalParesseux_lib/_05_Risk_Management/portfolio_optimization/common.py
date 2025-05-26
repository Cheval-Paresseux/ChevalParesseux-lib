import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from typing import Union, Self
import uuid



#! ==================================================================================== #
#! ====================================== Base Model ================================== #
class Portfolio(ABC):
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1,
    ) -> None:
        # ======= I. Jobs =======
        self.n_jobs = n_jobs

    #?____________________________________________________________________________________ #
    @abstractmethod
    def set_params(
        self,
        **kwargs
    ) -> Self:
        """
        Sets the parameter of the portfolio.

        Parameters:
            - **kwargs: additional parameters for the portfolio.

        Returns:
            - Self: The instance of the class with the parameters
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
        Preprocesses the data before portfolio extraction.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to be processed.
            - **kwargs: Additional parameters for the data processing.

        Returns:
            - tuple or pd.DataFrame or pd.Series: The processed data ready for portfolio extraction.
        """
        ...
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def predict(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
        **kwargs
    ) -> Union[list, pd.DataFrame]:
        """
        Core method for portfolio extraction.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to extract the signals from
            - **kwargs: Additional parameters for the signal extraction.
        
        Returns:
            - list or pd.DataFrame: The extracted signals.
        """
        ...
