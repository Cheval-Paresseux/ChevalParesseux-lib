import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class MetaModel(ABC):
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def set_params(self):
        pass

    @abstractmethod
    def extract(self):
        pass