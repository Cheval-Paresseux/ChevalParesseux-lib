from ..signal_processing import common as com

import numpy as np
import pandas as pd
from typing import Union, Self
 


#! ==================================================================================== #
#! ================================ Tree Classifiers ================================== #
class Volume_filter(com.SignalProcessor):
    #?____________________________________________________________________________________ #
    def __init__(
        self, 
        n_jobs: int = 1,
    ) -> None:
        # ======= I. Initialization ======= 
        super().__init__(n_jobs=n_jobs)
        
        self.params = {}
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        window: int,
    ) -> Self:
        
        self.params = {
            'window': window
        }
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: Union[tuple, pd.Series, pd.DataFrame],
    ) -> tuple:
        
        return data
    
    #?____________________________________________________________________________________ #
    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray], 
        y_train: Union[pd.Series, np.ndarray]
    ) -> Self:
        # ======= I. Data Processing ======= 
        
        # ======= II. Model Fitting =======
        
        # ======= III. Model Saving =======
        
        return self
    
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: pd.DataFrame
    ) -> pd.Series:
        
        return X_test

#*____________________________________________________________________________________ #