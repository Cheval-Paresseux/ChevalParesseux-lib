from ..sizing import common as com

import numpy as np
import pandas as pd
from typing import Union, Self
 


#! ==================================================================================== #
#! ================================ Risk Based Sizers ================================= #
class Volatility_sizer(com.Sizer):
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
        target_volatility: float,
        estimation_window: int,
        estimation_column: str,
        minimum_change: float = 0.01,
        max_leverage: float = 1.0,
    ) -> Self:
        # ======= I. Set Parameters =======
        self.params = {
            'target_volatility': target_volatility,
            'estimation_window': estimation_window,
            'estimation_column': estimation_column,
            'minimum_change': minimum_change,
            'max_leverage': max_leverage,
        }
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: pd.DataFrame,
    ) -> tuple:
        # ======= I. Parameters =======
        estimation_column = self.params['estimation_column']
        estimation_window = self.params['estimation_window']

        # ======= II. Estimate Volatility =======
        processed_data = data.copy()
        returns = processed_data[estimation_column].pct_change()
        volatility = returns.rolling(window=estimation_window).std()

        # ======= III. Manage output =======
        processed_data['volatility'] = volatility
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def fit(
        self, 
    ) -> Self:
        return self
    
    #?____________________________________________________________________________________ #
    def extract(
        self, 
        data: pd.DataFrame,
    ) -> pd.Series:
        # ======= I. Process data =======
        processed_data = self.process_data(data)

        # ======= II. Extract raw size =======
        target_volatility = self.params['target_volatility']
        processed_data['raw_size'] = target_volatility / processed_data['volatility']
        max_leverage = self.params['max_leverage']
        processed_data['raw_size'] = processed_data['raw_size'].round(2).clip(lower=0.0, upper=max_leverage)

        # ======= III. Initialize final_size =======
        processed_data['final_size'] = 0.0
        current_signal = 0
        current_size = 0.0

        for idx, row in processed_data.iterrows():
            signal = row.get('signal', 0)

            if signal == 0:
                current_signal = 0
                current_size = 0.0
                processed_data.at[idx, 'final_size'] = 0.0

            elif signal != current_signal:
                current_signal = signal
                current_size = row['raw_size']
                processed_data.at[idx, 'final_size'] = current_size

            else:  # signal == current_signal and != 0
                change_in_size = abs(row['raw_size'] - current_size)
                if change_in_size >= self.params['minimum_change']:
                    current_size = row['raw_size']

                processed_data.at[idx, 'final_size'] = current_size

        # ======= IV. Output =======
        size_series = processed_data['final_size']
        size_series.name = 'size'

        return size_series

#*____________________________________________________________________________________ #