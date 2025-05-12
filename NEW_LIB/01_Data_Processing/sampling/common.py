from ...utils import calculations as calc

import pandas as pd
import numpy as np
from typing import Union, Self
from abc import ABC, abstractmethod
from joblib import Parallel, delayed



#! ==================================================================================== #
#! =================================== Builders  ====================================== #
class DatasetBuilder(ABC):
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the DatasetBuilder object.

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
        Sets the parameter grid for the datset extraction.

        Parameters:
            - **kwargs: Each parameter should be a list of possible values.

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
        Preprocesses the data before building the dataset.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to be processed.
            - **kwargs: Additional parameters for the data processing.

        Returns:
            - tuple or pd.DataFrame or pd.Series: The processed data ready for dataset extraction.
        """
        ...
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_dataset(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
        **kwargs
    ) -> Union[list, pd.DataFrame]:
        """
        Core method for Dataset extraction.
        
        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to extract the dataset from
            - **kwargs: Additional parameters for the dataset extraction.
        
        Returns:
            - Union[list, pd.DataFrame] : The extracted dataset(s) as a pd.DataFrame or a list of pd.DataFrame.
        """
        ...
        
    #?_________________________________ Callable methods _________________________________ #
    def vertical_stacking(
        self, 
        dfs_list: list
    ) -> pd.DataFrame:
        """
        Applies vertical stacking to a list of DataFrames.
        
        Parameters:
            - dfs_list (list): List of DataFrames to be stacked.
        
        Returns:
            - stacked_data (pd.DataFrame): The vertically stacked DataFrame.
        """
       # ======= I. Ensure all DataFrames have the same columns =======
        if len(dfs_list) < 1:
            raise ValueError("The list does not contain enough DataFrames.")

        columns = dfs_list[0].columns
        for df in dfs_list[1:]:
            if not df.columns.equals(columns):
                raise ValueError("All DataFrames must have the same columns.")

        # ======= II. Concatenate DataFrames horizontally =======
        stacked_data = pd.concat(dfs_list, axis=0, ignore_index=True)
        
        return stacked_data
    
    #?____________________________________________________________________________________ #
    def extract(
        self, 
        data: Union[tuple, pd.Series, pd.DataFrame]
    ) -> list:
        """
        Main method to extract dataset.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to extract the datset from
        
        Returns:
            - datasets (list): List of DataFrames, each representing a dataset for a specific parameter combination.
        """
        # ======= I. Extract the Parameters Universe =======
        params_grid = calc.get_dict_universe(self.params)

        # ======= II. Extract the dataset for each Parameters =======
        datasets = Parallel(n_jobs=self.n_jobs)(delayed(self.get_datset)(data, **params) for params in params_grid)

        return datasets
