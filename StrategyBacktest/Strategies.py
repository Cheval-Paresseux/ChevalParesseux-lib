import sys
sys.path.append("../")
import Data as dt
import Features as ft

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

#! ==================================================================================== #
#! ====================================== Base Model ================================== #
class Strategy(ABC):
    def __init__(self):
        self.data = None
        self.processed_data = None
    
    #*____________________________________________________________________________________ #
    @abstractmethod
    def load_data(self):
        pass

    #*____________________________________________________________________________________ #
    @abstractmethod
    def set_params(self):
        pass

    #*____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self):
        pass
    
    #*____________________________________________________________________________________ #
    @abstractmethod
    def fit(self):
        pass
    
    #*____________________________________________________________________________________ #
    @abstractmethod
    def predict(self):
        pass

    #*____________________________________________________________________________________ #
    @abstractmethod
    def operate(self):
        pass



#! ==================================================================================== #
#! ====================================== Strategies ================================== #
class MA_crossover(Strategy):
    def __init__(self):
        super().__init__()

        self.window = None
    
    #*____________________________________________________________________________________ #
    def load_data(self, ticker: str, start_date: str, end_date: str):
        data = dt.load_data(ticker)
        data = data.loc[start_date:end_date]

        self.data = data

        return data
    
    #*____________________________________________________________________________________ #
    def process_data(self):
        data = self.data.copy()
        close_series = data['close']

        moving_average = ft.average_features(price_series=close_series, window=self.window)
        data['MA'] = moving_average

        processed_data = [data]

        self.processed_data = processed_data

        return processed_data
    
    #*____________________________________________________________________________________ #
    def set_params(self, window: int):
        self.window = window
    
    #*____________________________________________________________________________________ #
    def fit(self):
        pass

    #*____________________________________________________________________________________ #
    def predict(self, df: pd.DataFrame):
        signals_df = df.copy()
        signals_df['Signal'] = 0

        signals_df.loc[signals_df['MA'] < 0, 'Signal'] = 1
        signals_df.loc[signals_df['MA'] > 0, 'Signal'] = -1

        return signals_df
    
    #*____________________________________________________________________________________ #
    def operate(self):
        signals_df = self.predict(self.processed_data)
        
        return signals_df

        




