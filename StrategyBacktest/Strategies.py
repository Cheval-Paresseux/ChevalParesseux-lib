import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class Strategy(ABC):
    def __init__(self):
            pass
    
    @abstractmethod
    def transform_data(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def operate(self):
        pass
