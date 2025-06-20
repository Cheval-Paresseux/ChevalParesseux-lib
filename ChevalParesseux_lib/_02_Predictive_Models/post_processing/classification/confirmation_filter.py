from ...post_processing import common as com

import numpy as np
import pandas as pd
from typing import Union, Self
 


#! ==================================================================================== #
#! ============================== Signals Based Filters =============================== #
class Confirmation_processor(com.SignalProcessor):
    """
    Class for applying a confirmation filter to time series data.
    
    This filter is used to smooth out the signals in order to avoid overtrading, by waiting
    for a certain number of confirmations before executing a trade.
    """
    #?____________________________________________________________________________________ #
    def __init__(
        self, 
        n_jobs: int = 1,
    ) -> None:
        """
        Constructor for the Confirmation_filter class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        # ======= I. Initialization ======= 
        super().__init__(n_jobs=n_jobs)
        
        self.params = {}
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        nb_confirmations: int,
    ) -> Self:
        """
        Sets the parameters for the confirmation filter.
        
        Parameters:
            - nb_confirmations (int): Number of confirmations required to execute a trade.
        
        Returns:
            - Self: The instance of the class with the parameter set.
        """
        # ======= I. Set Parameters =======
        self.params = {
            'nb_confirmations': nb_confirmations
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
    ) -> Self:
        """
        No fitting required for this filter.
        """
        return self
    
    #?____________________________________________________________________________________ #
    def extract(
        self, 
        signal_series: pd.Series
    ) -> pd.Series:
        """
        Transforms the test data using the confirmation filter.
        
        Parameters:
            - X_test (pd.Series): The test data to be transformed.
        
        Returns:
            - pd.Series: The transformed test data.
        """
        # ======= I. Parameters =======
        nb_confirmations = self.params['nb_confirmations']
        
        # ======= II. Apply Confirmation Filter =======
        processed_signals = []  
        
        current_signal = signal_series.iloc[0]  
        count = 0 
        for signal in signal_series:
            if signal == current_signal:
                count = 0  # Reset counter if the signal remains the same
            else:
                count += 1  # Increment counter on change

            if count > nb_confirmations:  # Confirm change if threshold is met
                current_signal = signal
                count = 0  # Reset counter

            processed_signals.append(current_signal)
        
        # ======= III. Manage output =======
        processed_signals = pd.Series(processed_signals, index=signal_series.index)
        processed_signals.name = signal_series.name
        
        return processed_signals

#*____________________________________________________________________________________ #