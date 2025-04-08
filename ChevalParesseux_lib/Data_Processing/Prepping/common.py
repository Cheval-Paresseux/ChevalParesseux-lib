from ..Prepping import sanitizing as sanit
from ..Prepping import sampling as sampl

import pandas as pd 
import numpy as np
from typing import Union
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ==================================== Cleaning ====================================== #
class DataCleaner():
    def __init__(
        self, 
        training_data: Union[list, pd.DataFrame],
        non_feature_columns: list = ['open', 'high', 'low', 'close', 'volume'],
        n_jobs: int = 1
    ):
        # ======= II. Store the inputs =======
        self.training_data = training_data
        self.non_feature_columns = non_feature_columns
        self.n_jobs = n_jobs

        # ======= III. Initialize Results =======
        self.params = None
        self.stacked_data = None
        self.processed_data = None
        self.features_informations = None
    
    #?__________________________________________________________________________________ #
    def set_params(
        self,
        stationarity_threshold: float = 0.05,
        outliers_threshold: float = 3,
        mean_tolerance: float = 0.01,
        std_tolerance: float = 0.01
    ):
        """
        This function sets the parameters for the DataCleaner class.
        Parameters:
            - stationarity_threshold (float): The threshold for stationarity check.
            - outliers_threshold (float): The threshold for outlier detection.
            - mean_tolerance (float): The tolerance for mean check.
            - std_tolerance (float): The tolerance for standard deviation check.
        """
        self.params = {
            'stationarity_threshold': stationarity_threshold,
            'outliers_threshold': outliers_threshold,
            'mean_tolerance': mean_tolerance,
            'std_tolerance': std_tolerance
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def vertical_stacking(self, dfs_list: list):
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
    def check_features_df(self, training_df: pd.DataFrame):
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
        # ======= I. Ensure the Features_df only contains features =======
        columns_name = training_df.columns.tolist()
        features_list = [column for column in columns_name if column not in self.non_feature_columns]
        
        features_df = training_df[features_list].copy()
        
        # ======= II. Drop Columns with only nans =======
        features_df = features_df.dropna(axis=1, how='all')

        # ======= II. Performs the Features Check =======
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(sanit.check_feature)(
                feature_series=features_df[feature], 
                stationarity_threshold=self.params['stationarity_threshold'], 
                outliers_threshold=self.params['outliers_threshold'], 
                mean_tolerance=self.params['mean_tolerance'], 
                std_tolerance=self.params['std_tolerance'], 
            ) for feature in features_df.columns
        )
        
        # ======= II. Recreate the Feature DataFrame =======
        scaled_data = pd.concat([result[0] for result in results], axis=1)
        features_informations = pd.concat([result[1] for result in results], axis=0)
        
        return scaled_data, features_informations
        
    #?_____________________________ User Functions _______________________________________ #
    def extract(self):
        """
        This function prepares the training data for the model by checking the features and stacking the DataFrames.
        """
        if isinstance(self.training_data, list):
            # ======= I. Check Features for all days =======
            features_informations = []
            processed_data = []
            idx = 0
            for individual_df in self.training_data:
                scaled_features, features_infos = self.check_features_df(training_df=individual_df)
                data = individual_df[self.non_feature_columns].copy()
                data = pd.concat([data, scaled_features], axis=1)
                data = data.dropna(axis=0)
                
                features_infos['df_index'] = idx
                idx += 1
                
                processed_data.append(data)
                features_informations.append(features_infos)

            # ======= II. Stack the DataFrames =======
            stacked_data = self.vertical_stacking(dfs_list=processed_data)
            features_informations = pd.concat(features_informations, axis=0, ignore_index=True)
            
            # ======= III. Store the Results =======
            self.processed_data = processed_data
            self.features_informations = features_informations
            self.stacked_data = stacked_data
            
        else:
            # ======= I. Check Features =======
            scaled_features, features_informations = self.check_features_df(features_df=self.training_data)
            data = individual_df[self.non_feature_columns].copy()
            data = pd.concat([data, scaled_features], axis=1)
            data = data.dropna(axis=0)
            
            # ======= II. Store the Results =======
            self.processed_data = data
            self.features_informations = features_informations
            self.stacked_data = data.copy()
            
        return stacked_data, processed_data, features_informations


#! ==================================================================================== #
#! =================================== Sampling  ====================================== #
def extract_targetBars(
    data: pd.DataFrame, 
    target_bars: int,
    column_name: str = "volume",
    window_bars_estimation: int = 10,
    new_cols_methods: str = "mean",
    grouping_column: str = "date",
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    This function performs a resampling of the DataFrame based on the cumulative sum of a specified column.
    It estimates the threshold for each DataFrame based on the average of previous days, the first day is set to the target_bars.
    Parameters:
        - data (pd.DataFrame) : The DataFrame to be resampled.
        - target_bars (int) : The target number of bars.
        - column_name (str) : The column name to calculate the cumulative sum. Default is "volume".
        - window_bars_estimation (int) : The number of previous days to consider for estimating the average. Default is 10.
        - new_cols_methods (str) : The method to aggregate additional columns. Default is "mean".
        - grouping_column (str) : The column name to group by. Default is "date".
        - n_jobs (int) : The number of parallel jobs to run. Default is 1.
    
    Returns:
        - Resampled DataFrame.
    """
    # ======= I. Group the DataFrame if Necessary =======
    if grouping_column is not None:
        dfs_list = sampl.get_groups_list(data=data, column_name=grouping_column)
    else:
        dfs_list = [data]
    
    # ======= II. Extract the thresholds =======
    thresholds = [target_bars] # The first threshold is the target_bars as we don't have previous days to estimate the average.
    for idx in range(1, len(dfs_list)):
        if idx < window_bars_estimation:
            previous_days = dfs_list[:idx]
        else:
            previous_days = dfs_list[idx - window_bars_estimation : idx]

        average = np.mean([day[column_name].sum() for day in previous_days])
        threshold = int(average / target_bars)
        thresholds.append(threshold)
    
    # ======= III. Process each DataFrame in parallel or sequentially =======
    if n_jobs == 1:
        resampled_dfs = [sampl.get_cumsum_resample(df=dfs_list[idx], column_name=column_name, threshold=thresholds[idx], new_cols_method=new_cols_methods) for idx in range(len(dfs_list))]
    else:
        resampled_dfs = Parallel(n_jobs=n_jobs)(
            delayed(sampl.get_cumsum_resample)(df=dfs_list[idx], column_name=column_name, threshold=thresholds[idx], new_cols_method=new_cols_methods)
            for idx in range(len(dfs_list))
    )

    return resampled_dfs
