from ..features_selection import common as com

import numpy as np 
import pandas as pd
from typing import Self


#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class Correlation_selector(com.FeaturesSelector):
    """
    Feature selection based on correlation.
    
    This class implements a feature selection method that removes features with high correlation
    to other features. It computes the correlation matrix and drops features that exceed a
    specified correlation threshold.
    """
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(
        self,
        n_jobs: int = 1
    ) -> None:
        """
        Constructor for the Correlation_selector class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use during feature computation.
        """
        # ======= I. Initialize Class =======
        super().__init__(n_jobs=n_jobs)
        
        # ======= II. Initialize Auxilaries =======
        self.features_to_drop = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        correlation_threshold: float = 0.9
    ) -> Self:
        """
        Sets the parameter grid for the feature extraction.
        
        Parameters:
            - correlation_threshold (float): The threshold for feature correlation.
        
        Returns:
            - Self: The instance of the class with the parameter grid set.
        """
        self.params = {
            'correlation_threshold': correlation_threshold
        }
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes the input data and returns a DataFrame.
        
        Parameters:
            - data (pd.DataFrame): The input data to be processed.
        
        Returns:
            - pd.DataFrame: A DataFrame containing the processed data.
        """
        return data
    
    #?____________________________________________________________________________________ #
    def fit(
        self, 
        data: pd.DataFrame
    ) -> Self:
        """
        Fit the model to the data.
        
        Parameters:
            - data (pd.DataFrame): The input data to fit the model.
        
        Returns:
            - Self: The instance of the class with the fitted model.
        """
        # ======= I. Compute Correlaiton Matrix =======
        correlation_threshold = self.params['correlation_threshold']
        corr_matrix = data.corr().abs()

        # ======= II. Extract features with too high correlation =======
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    to_drop.add(corr_matrix.columns[j])  
        
        self.features_to_drop = to_drop
        
        return self
    
    #?____________________________________________________________________________________ #
    def extract(
        self, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract features from the given data.
        
        Parameters:
            - data (pd.DataFrame): The input data from which features are to be extracted.
        
        Returns:
            - pd.DataFrame: The filtered data with selected features.
        """
        filtered_data = data.drop(columns=self.features_to_drop).copy()

        return filtered_data
    
    