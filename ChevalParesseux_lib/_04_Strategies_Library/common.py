import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from typing import Union, Self
import uuid



#! ==================================================================================== #
#! ====================================== Base Model ================================== #
class Strategy(ABC):
    """
    Base class for all strategies.
    
    This class is used to define the common methods and attributes for all strategies.
    Its main purpose is to provide the operate method to extract the operations from the signals.
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1,
    ) -> None:
        """
        """
        # ======= I. Jobs =======
        self.n_jobs = n_jobs

    #?____________________________________________________________________________________ #
    @abstractmethod
    def set_params(
        self,
        **kwargs
    ) -> Self:
        """
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
        """
        ...
