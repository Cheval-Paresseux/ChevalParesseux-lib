from ..Tuning import common as com

import numpy as np 
import pandas as pd


#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class NoCorrelationSelector(com.FeaturesSelector):
    def __init__(
        self
    ):
        self.params = None
        self.to_drop = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        correlation_threshold: float = 0.9
    ):
        self.params = {
            'correlation_threshold': correlation_threshold
        }
        
        return self
    
    #?____________________________________________________________________________________ #
    def extract(
        self, 
        features_df: pd.DataFrame
    ):
        # ======= I. Compute Correlaiton Matrix =======
        correlation_threshold = self.params['correlation_threshold']
        corr_matrix = features_df.corr().abs()

        # ======= II. Extract features with too high correlation =======
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    to_drop.add(corr_matrix.columns[j])  
        
        self.to_drop = to_drop
        
        # ======= III. Drop features with too high correlation =======
        filtered_features = features_df.drop(columns=self.to_drop).copy()
        
        return filtered_features
    
    #?____________________________________________________________________________________ #
    def extract_new(
        self, 
        features_df: pd.DataFrame
    ):

        filtered_features = features_df.drop(columns=self.to_drop).copy()

        return filtered_features
    
    