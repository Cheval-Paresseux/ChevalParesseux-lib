from ..Prepping import data_verification as verif

import pandas as pd
import numpy as np
from typing import Union
from joblib import Parallel, delayed


#! ==================================================================================== #
#! ==================================== Preparator ==================================== #
class TrainingPreparator():
    def __init__(self, training_data: Union[list, pd.DataFrame]):
        # ======= I. Store the training data =======
        self.training_data = training_data
        self.stacked_data = None
        
        # ======= II. Set default parameters =======
        self.stationarity_threshold = None
        self.outliers_threshold = None
        self.mean_tolerance = None
        self.std_tolerance = None
        self.n_jobs = 1
        
        # ======= III. Post Processing Information =======
        self.processed_data = None
        self.features_informations = None
        
        # ======= IV. Non-Feature Columns =======
        self.non_feature_columns = None
    
    #?_____________________________ Build Functions ______________________________________ #
    def horizontal_stacking(self, dfs_list: list):
        """
        This function stacks the DataFrames horizontally.
        Parameters:
            - dfs_list (list): A list of DataFrames to be stacked.
            
        Returns:
            - stacked_data (pd.DataFrame): The stacked DataFrame.
        """
       # ======= I. Ensure all DataFrames have the same columns =======
        if len(dfs_list) < 1:
            raise ValueError("The list does not contain enough DataFrames.")

        columns = dfs_list[0].columns
        for df in dfs_list[1:]:
            if not df.columns.equals(columns):
                raise ValueError("All DataFrames must have the same columns.")

        # ======= II. Concatenate DataFrames horizontally =======
        stacked_data = pd.concat(dfs_list, axis=0, ignore_index=True)
        self.stacked_data = stacked_data.copy()
        
        return stacked_data
        
    #?____________________________________________________________________________________ #
    def check_features_df(self, features_df: pd.DataFrame):
        """
        This function checks the features of the DataFrame to ensure they are properly formatted and valid.
        It performs the following checks:
            - Stationarity check
            - Outlier detection
            - Mean and standard deviation checks
        
        It returns the processed features DataFrame and the features information DataFrame.
        Parameters:
            - features_df (pd.DataFrame): The DataFrame containing the features to be checked.
        
        Returns:
            - scaled_data (pd.DataFrame): The processed features DataFrame.
            - features_informations (pd.DataFrame): The DataFrame containing the features information.
        """
        # ======= I. Ensure the Features_df does not contain the labels =======
        columns_name = features_df.columns.tolist()
        features_list = [column for column in columns_name if column not in self.non_feature_columns]
        
        self.features_df = features_df.copy()
        self.features_df = self.features_df[features_list]

        # ======= II. Performs the Features Check =======
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(verif.check_feature)(
                feature_series=self.features_df[feature], 
                stationarity_threshold=self.stationarity_threshold, 
                outliers_threshold=self.outliers_threshold, 
                mean_tolerance=self.mean_tolerance, 
                std_tolerance=self.std_tolerance, 
            ) for feature in self.features_df.columns
        )
        
        # ======= II. Recreate the Feature DataFrame =======
        scaled_data = pd.concat([result[0] for result in results], axis=1)
        features_informations = pd.concat([result[1] for result in results], axis=0)
        
        return scaled_data, features_informations
        
    #?_____________________________ User Functions _______________________________________ #
    def set_non_features_columns(self, non_feature_columns: list = ['open', 'high', 'low', 'close', 'volume']):
        self.non_feature_columns = non_feature_columns
        
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        stationarity_threshold: float = 0.05, 
        outliers_threshold: float = 3, 
        mean_tolerance: float = 0.01, 
        std_tolerance: float = 0.01, 
        n_jobs: int = 1
    ):
        """
        This function sets the parameters for the training preparator.
        Parameters:
            - stationarity_threshold (float): The threshold for stationarity tests.
            - outliers_threshold (float): The threshold for outlier detection.
            - mean_tolerance (float): The tolerance for the mean value.
            - std_tolerance (float): The tolerance for the standard deviation.
            - n_jobs (int): The number of jobs to run in parallel.
        """
        self.stationarity_threshold = stationarity_threshold
        self.outliers_threshold = outliers_threshold
        self.mean_tolerance = mean_tolerance
        self.std_tolerance = std_tolerance
        self.n_jobs = n_jobs
    
    #?____________________________________________________________________________________ #
    def prepare_data(self):
        """
        This function prepares the training data for the model by checking the features and stacking the DataFrames.
        """
        if isinstance(self.training_data, list):
            # ======= I. Check Features for all days =======
            features_informations = []
            processed_data = []
            idx = 0
            for day_df in self.training_data:
                scaled_features, features_infos = self.check_features_df(features_df=day_df)
                data = day_df[self.non_feature_columns].copy()
                data = pd.concat([data, scaled_features], axis=1)
                data = data.dropna(axis=0)
                
                features_infos['day'] = idx
                idx += 1
                
                processed_data.append(data)
                features_informations.append(features_infos)

            # ======= II. Stack the DataFrames =======
            stacked_data = self.horizontal_stacking(dfs_list=processed_data)
            features_informations = pd.concat(features_informations, axis=0, ignore_index=True)
            
            # ======= III. Store the Results =======
            self.processed_data = processed_data
            self.features_informations = features_informations
            self.stacked_data = stacked_data
            
        else:
            # ======= I. Check Features =======
            scaled_features, features_informations = self.check_features_df(features_df=self.training_data)
            data = day_df[self.non_feature_columns].copy()
            data = pd.concat([data, scaled_features], axis=1)
            data = data.dropna(axis=0)
            
            # ======= II. Store the Results =======
            self.processed_data = data
            self.features_informations = features_informations
            self.stacked_data = data.copy()
            
        return stacked_data, processed_data, features_informations

#*____________________________________________________________________________________ #