from ..Meta_Models import common as com

import numpy as np
import pandas as pd

class noFilter(com.MetaModel):
    def __init__(self):
        super().__init__()

    def set_params(self):
        return self
    
    def extract(self, predictions: pd.Series):
        return predictions