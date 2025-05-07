from ...utils import function_tools as tools
from ...Data_Processing.Measures import Filters as fil

import numpy as np
import pandas as pd
from typing import Union
from typing import Self
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class PricingModel(ABC):
    """
    Abstract base class for all pricing models.
    
    This class defines the core structure and interface for pricers. It is meant to be subclassed
    by specific pricing implementations. 
    Subclasses must implement the following abstract methods:
        - __init__: Initializes the portfolio with number of jobs.
        - set_params: Defines the parameter grid as a dictionary of lists.
        - process_data: Applies preprocessing to the data.
        - get_price: Extracts the actual price(s).

    Main usage involves one core methods:
        - extract: Returns extracted prices.
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1
    ):
        """
        Constructor for the PricingModel class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use during feature computation.
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
        Sets the parameter grid for the price extraction.

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
        data: Union[tuple, pd.DataFrame],
        **kwargs
    ) -> Union[tuple, pd.DataFrame, pd.Series]:
        """
        Preprocesses the data before price extraction.

        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
            - **kwargs: Additional parameters for the data processing.

        Returns:
            - tuple or pd.DataFrame or pd.Series: The processed data ready for price extraction.
        """
        ...
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_price(
        self,
        data: Union[tuple, pd.DataFrame],
        **kwargs
    ) -> pd.DataFrame:
        """
        Core method for price extraction.
        
        Parameters:
            - data (tuple | pd.DataFrame): The input data to extract the price from.
            - **kwargs: Additional parameters for the price extraction.
        
        Returns:
            - pd.DataFrame : The extracted price as a pd.DataFrame.
        """
        ...
       
    #?____________________________________________________________________________________ #
    def extract(
        self, 
        data: Union[tuple, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Main method to extract prices.

        Parameters:
            - data (tuple | pd.DataFrame): The input data to extract the price from.
        
        Returns:
            - portfolio_df (pd.DataFrame): The extracted prices as a DataFrame.
        """
        # ======= I. Extract the Parameters Universe =======
        params_grid = tools.get_dict_universe(self.params)

        # ======= II. Extract the features for each Parameters =======
        portfolios = Parallel(n_jobs=self.n_jobs)(delayed(self.get_feature)(data, **params) for params in params_grid)

        return portfolios
