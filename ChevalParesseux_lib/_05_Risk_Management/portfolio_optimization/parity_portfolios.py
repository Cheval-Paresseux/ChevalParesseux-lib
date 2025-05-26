from ..portfolio_optimization import common as com 

import pandas as pd
import numpy as np
from typing import Union, Self
from joblib import Parallel, delayed



#! ==================================================================================== #
#! ====================================== Base Model ================================== #
class Equalweights_portfolio(com.Portfolio):
    """
    Class for creating an equal weights portfolio.
    
    It inherits from the Portfolio class and implements methods to process data and extract equal weights for each asset in the portfolio.
    The portfolio weights are assigned equally to all assets in the portfolio.
    """
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(
        self, 
        n_jobs: int = 1,
    ) -> None:
        """
        Constructor for the Equalweights_portfolio class.
        
        Parameters:
            - n_jobs (int): The number of jobs to run in parallel, default is 1.
        """
        # ======= I. Jobs =======
        self.n_jobs = n_jobs

    #?____________________________________________________________________________________ #
    def set_params(self) -> Self:
        """
        Sets the parameters of the portfolio.
        This method is currently a placeholder and does not set any parameters.
        """
        return self
    
    #?________________________________ Auxiliary methods _________________________________ #
    def process_data(
        self,
        data: Union[tuple, list, pd.DataFrame],
    ) -> list:
        """
        Preprocesses the data before portfolio extraction.
        
        Parameters:
            - data (list | pd.DataFrame): The input data to be processed, should be a list of DataFrames or a single DataFrame.
        
        Returns:
            - list: The processed data ready for portfolio extraction, returns a list of DataFrames.
        """
        if isinstance(data, pd.DataFrame): # If data is a single DataFrame, convert it to a list
            processed_data = [data]
        
        elif isinstance(data, tuple): # If data is a tuple, convert it to a list and check if it contains DataFrames
            processed_data = list(data)
            if not all(isinstance(df, pd.DataFrame) for df in processed_data):
                raise ValueError("All elements in the tuple must be DataFrames.")
            
        elif isinstance(data, list): # If data is a list, check if it contains DataFrames
            processed_data = data
            if not all(isinstance(df, pd.DataFrame) for df in processed_data):
                raise ValueError("All elements in the list must be DataFrames.")
        
        else:
            raise ValueError("Data must be a DataFrame, list of DataFrames, or tuple of DataFrames.")
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def extract(
        self,
        data: Union[tuple, list, pd.DataFrame],
    ) -> list:
        """
        Extracts the equal weights portfolio from the given data.
        
        Parameters:
            - data (tuple | list | pd.DataFrame): The input data as a DataFrame or list of DataFrames, the dataframes should be the sigals_df from a strategy.
            
        Returns:
            - list : The list of signals_df with a new column 'portfolio_weights' containing the equal weights for each asset.
        """
        # ======= I. Process data =======
        processed_data = self.process_data(data)
        
        # ======= II. Extract portfolio weights =======
        n_assets = len(processed_data)
        weight = 1 / n_assets
        
        # ======= III. Assign weights to each asset =======
        signals_dfs = []
        for df in processed_data:
            df['portfolio_weight'] = weight
            signals_dfs.append(df)
        
        return signals_dfs
    
    