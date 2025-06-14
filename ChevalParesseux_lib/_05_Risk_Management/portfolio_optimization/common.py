import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from typing import Union, Self
import uuid



#! ==================================================================================== #
#! ====================================== Base Model ================================== #
class Portfolio(ABC):
    """
    Base class for all portfolios.
    
    This class is used to define the common methods and attributes for all portfolios.
    Each portfolio should inherit from this class and implement the abstract methods. Also, they should
    always have a similar output format, which is a list of DataFrames where each DataFrame corresponds
    to the 'signals_df' of an asset (contains basic asset information, signals, size and portfolio weight information).
    signals_df is a DataFrame that contains the following columns:
        - 'code': Unique identifier for the asset.
        - 'signal': The signal generated by the portfolio.
        - 'size': The size of the position to take based on the signal.
        - 'portfolio_weight': The weight of the asset in the portfolio.
        - 'date': timestamp of the signal.
        - A price column (e.g., 'close', 'open', 'ask_open', etc.) that contains the price of the asset at the time of the signal.
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1,
    ) -> None:
        """
        Constructor for the Portfolio class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to run. Default is 1 (no parallelization).
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
        Sets the parameter of the portfolio.

        This method should be implemented by each portfolio to set its specific parameters.
        Usually, we want the parameters to be feed independenly inside the portfolio and store them in a 
        dictionary or similar structure.
        We specifically want to avoid losing tracks of which parameters are to be used for which portfolio.
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

        This method is used to prepare the input data for the portfolio extraction process.
        It can include tasks such as normalization, feature engineering, or any other preprocessing steps.
        
        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to preprocess.
            - **kwargs: Additional parameters for preprocessing.
        
        Returns:
            - Union[tuple, pd.DataFrame, pd.Series]: The preprocessed data.
        """
        ...
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def extract(
        self,
        data: Union[list, pd.DataFrame],
        **kwargs
    ) -> Union[list, pd.DataFrame]:
        """
        Core method for portfolio extraction.

        This method should be implemented by each portfolio to extract signals from the input data.
        It should return a list of DataFrames or a single DataFrame, where each DataFrame corresponds to the 'signals_df' of an asset.
        
        The DataFrame should contain the following columns:
            - 'code': Unique identifier for the asset.
            - 'signal': The signal generated by the portfolio.
            - 'size': The size of the position to take based on the signal.
            - 'portfolio_weight': The weight of the asset in the portfolio.
            - 'date': Timestamp of the signal.
            - A price column (e.g., 'close', 'open', 'ask_open', etc.) that contains the price of the asset at the time of the signal.
        
        Parameters:
            - data (list | pd.DataFrame): The input data to extract signals from.
            - **kwargs: Additional parameters for extraction.
        
        Returns:
            - Union[list, pd.DataFrame]: The extracted signals as a list of DataFrames or a single DataFrame.
        """
        ...
