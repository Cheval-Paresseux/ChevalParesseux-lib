import time
import numpy as np
import pandas as pd
import plotly.express as px
from abc import ABC, abstractmethod
from joblib import Parallel, delayed


#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class strategy(ABC):
    """
    Abstract class for HFT strategies
    """

    @abstractmethod
    def set_params(self, params: dict):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform_data(df: pd.DataFrame, date: str):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        """
        Predict returns a dataframe with each row containing an operation. It's columns are as follows:
        - ID: The ID of the operation
        - Trade Type: The type of the operation (buy or sell)
        - Entry ts: The timestamp of the entry
        - Entry price: The price of the entry
        - Exit ts: The timestamp of the exit
        - Exit price: The price of the exit
        - Size: The size of the operation
        - Profit: The profit of the operation
        """
        pass

