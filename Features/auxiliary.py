import numpy as np
import pandas as pd

def get_minimum(series: pd.Series):
    minimum = np.min(series)
    
    return minimum

def get_maximum(series: pd.Series):
    maximum = np.max(series)
    
    return maximum 