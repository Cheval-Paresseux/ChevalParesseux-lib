from ..Prepping import sanitizing as sanit
from ..Prepping import sampling as sampl

import pandas as pd
import numpy as np
from typing import Union
from joblib import Parallel, delayed
import inspect


#! ==================================================================================== #
#! ==================================== Cleaning ====================================== #
class FeaturesCleaner():
    """
    Cleaner class for preprocessing time series features data.
    
    This class is designed to handle the following tasks:
        - Handling error features (e.g., NaN, infinite, constant columns)
        - Checking the features for marginal errors, outliers, scale and stationarity
        - Vertical stacking of DataFrames
        - Extracting rules for feature processing on test data
        - Extracting clean features from new data based on the rules defined during training
    """
    def __init__(
        self, 
        training_data: Union[list, pd.DataFrame],
        non_feature_columns: list = ['open', 'high', 'low', 'close', 'volume'],
        n_jobs: int = 1
    ):
        """
        Initializes the FeaturesCleaner class.
        
        Parameters:
            - training_data (Union[list, pd.DataFrame]): The training data to be processed.
            - non_feature_columns (list): List of columns that are not features.
            - n_jobs (int): Number of jobs to run in parallel.
        """
        # ======= II. Store the inputs =======
        self.training_data = training_data
        self.non_feature_columns = non_feature_columns
        self.n_jobs = n_jobs

        # ======= III. Initialize Results =======
        self.params = None
        self.stacked_data = None
        self.processed_data = None
        self.features_informations = None
        self.features_rules = None
        self.error_features = []
    
    #?__________________________________________________________________________________ #
    def set_params(
        self,
        stationarity_threshold: float = 0.05,
        outliers_threshold: float = 5,
    ):
        """
        Defines the parameters for the feature cleaning process.
        
        Parameters:
            - stationarity_threshold (float): The threshold for stationarity check.
            - outliers_threshold (float): The threshold for outlier detection.
        """
        self.params = {
            'stationarity_threshold': stationarity_threshold,
            'outliers_threshold': outliers_threshold,
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def check_error_features(
        self,
        df_list: list
    ) -> list:
        """
        This function checks for error features in the DataFrame list.
        
        Parameters:
            - df_list (list): List of DataFrames to be checked.
        
        Returns:
            - error_features (list): List of error features found in the DataFrames.
        """
        error_features = []
        for individual_df in df_list:
            # ====== I. Extract only features ======
            tested_df = individual_df.drop(columns=self.non_feature_columns, axis=1).copy()
            
            # ====== II. Check only nans columns & only infinite columns ======
            tested_df = tested_df.replace([np.inf, -np.inf], np.nan)
            only_nans_columns = tested_df.columns[tested_df.isna().all()].to_list()
            
            # ====== III. Check constant columns ======
            constant_columns = [col for col in tested_df.columns if tested_df[col].nunique(dropna=True) <= 1]
            
            # ====== IV. Check unstable features ======
            for feature in tested_df.columns:
                mean = tested_df[feature].mean()
                std = tested_df[feature].std()
                if std > 10 or mean > 10:
                    error_features.append(feature)
            
            # ====== IV. Store the errors features ======            
            error_features = error_features + only_nans_columns + constant_columns
        
        # Remove duplicates
        unique_error_features = list(set(error_features))
        
        return unique_error_features
    
    #?____________________________________________________________________________________ #
    def check_features_df(
        self, 
        training_df: pd.DataFrame
    ) -> tuple:
        """
        This function checks and clean the features of the DataFrame to ensure they are properly formatted and valid.
        It performs the following checks: Stationarity check | Outlier detection | Mean and standard deviation checks.
        
        Parameters:
            - features_df (pd.DataFrame): The DataFrame containing the features to be checked.
        
        Returns:
            - scaled_data (pd.DataFrame): The processed features DataFrame.
            - features_informations (pd.DataFrame): The DataFrame containing the features information.
        """
        # ======= I. Ensure the Features_df only contains features =======
        columns_name = training_df.columns.tolist()
        features_list = [column for column in columns_name if column not in self.non_feature_columns]
        features_list = [feature for feature in features_list if feature not in self.error_features]
        
        features_df = training_df[features_list].copy()
        
        # ======= II. Performs the Features Check =======
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(sanit.check_feature)(
                feature_series=features_df[feature], 
                stationarity_threshold=self.params['stationarity_threshold'], 
                outliers_threshold=self.params['outliers_threshold'],
            ) for feature in features_df.columns
        )
        
        # ======= II. Recreate the Feature DataFrame =======
        scaled_data = pd.concat([result[0] for result in results], axis=1)
        features_informations = pd.concat([result[1] for result in results], axis=0)
        
        return scaled_data, features_informations
    
    #?____________________________________________________________________________________ #
    def vertical_stacking(
        self, 
        dfs_list: list
    ) -> pd.DataFrame:
        """
        Applies vertical stacking to a list of DataFrames.
        
        Parameters:
            - dfs_list (list): List of DataFrames to be stacked.
        
        Returns:
            - stacked_data (pd.DataFrame): The vertically stacked DataFrame.
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
    def get_rules(
        self, 
        features_informations
    ) -> dict:
        """
        Extracts the rules for feature processing based on the features information DataFrame.
        
        Parameters:
            - features_informations (pd.DataFrame): The DataFrame containing the features information.
        
        Returns:
            - features_rules (dict): A dictionary containing the rules for each feature.
        """
        # ======= I. Group the Features Information =======
        grouped_infos = features_informations.groupby("feature_name")
        grouped_infos = [grouped_infos.get_group(x) for x in grouped_infos.groups]

        # ======== II. Create the Rules Dictionary =======
        features_rules = {}
        for feature in grouped_infos:
            # II.1 Extract the Feature Name and the Rules
            feature_name = feature["feature_name"].values[0]
            mean = feature["mean"].mean()
            std = feature["std"].mean()
            outliers_threhsold = feature["outliers_threshold"].mean()
            
            # II.3 Create the Rules Dictionary
            features_rules[feature_name] = {
                "mean": mean,
                "std": std,
                "outliers_threshold": outliers_threhsold,
            }
        
        # ======= III. Store the Results =======
        self.features_rules = features_rules
        
        return features_rules
    
    #?_____________________________ User Functions _______________________________________ #
    def extract(self):
        """
        Cleans the training data by performing the following steps:
            1. Check for error features (e.g., NaN, infinite, constant columns)
            2. Check the features for marginal errors, outliers, scale and stationarity
            3. Vertical stacking of DataFrames
            4. Extracting rules for feature processing on test data
        """
        # ======= I. Verify the input type =======
        if not isinstance(self.training_data, list):
            training_data = [self.training_data]
        else:
            training_data = self.training_data.copy()
            
        # ======= II Check for Error Features =======
        error_features = self.check_error_features(df_list=training_data)
        self.error_features = error_features
        
        # ======= III. Check the Features DataFrames =======
        features_informations = []
        processed_data = []
        idx = 0
        for individual_df in training_data:
            tested_df = individual_df.drop(columns=self.non_feature_columns, axis=1).copy()
            # III.1 Drop the error features
            if len(self.error_features) > 0:
                tested_df = tested_df.drop(columns=self.error_features, axis=1)

            # III.2 Check the features
            scaled_features, features_infos = self.check_features_df(training_df=tested_df)
            # III.3 Rearrange the final DataFrame
            non_feature_df = individual_df[self.non_feature_columns].copy()
            clean_df = pd.concat([non_feature_df, scaled_features], axis=1)
            clean_df = clean_df.ffill()
            clean_df = clean_df.dropna(axis=0)
            # III.3 Add the df index to the features information
            features_infos['df_index'] = idx
            idx += 1
            
            processed_data.append(clean_df)
            features_informations.append(features_infos)

        # ======= II. Stack the DataFrames =======
        stacked_data = self.vertical_stacking(dfs_list=processed_data)
        features_informations = pd.concat(features_informations, axis=0, ignore_index=True)
        features_rules = self.get_rules(features_informations)
        
        # ======= III. Store Results =======
        self.features_informations = features_informations
        self.features_rules = features_rules
        self.processed_data = processed_data
        self.stacked_data = stacked_data
        
        return stacked_data, processed_data, features_informations, features_rules

    #?____________________________________________________________________________________ #
    def extract_new(
        self, 
        new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extracts clean features from new data based on the rules defined during training.
        
        Parameters:
            - new_data (pd.DataFrame): The new data to be processed.
        
        Returns:
            - data (pd.DataFrame): The processed DataFrame with clean features.
        """
        def check_feature(feature_series: pd.Series, feature_name: str) -> pd.Series:
            # ======= I. Check for error values (nans & infs) =======
            clean_series, _, _, _, _ = sanit.check_for_error_values(feature_series=feature_series)
            
            # ======= II. Check for outliers =======
            feature_rules = self.features_rules[feature_name]
            outliers_threshold = feature_rules["outliers_threshold"]
            
            outliers_idxs = clean_series[abs(clean_series) > outliers_threshold].index
            filtered_series = clean_series.copy()
            filtered_series.loc[outliers_idxs] = np.nan 
            
            # ======= III. Check for Scale =======
            mean = feature_rules["mean"]
            std = feature_rules["std"]
            
            scaled_series = (filtered_series - mean) / std
            scaled_series.name = feature_name
            
            return scaled_series
        
        # ======= I. Ensure the Features_df only contains features =======
        new_df = new_data.copy()
        columns_name = new_df.columns.tolist()
        features_list = [column for column in columns_name if column not in self.non_feature_columns]
        features_list = [feature for feature in features_list if feature not in self.error_features]
        
        features_df = new_df[features_list].copy()
        
        # ======= II. Extract the clean features =======
        clean_features_list = []
        for feature in features_df.columns:
            clean_series = check_feature(feature_series=features_df[feature], feature_name=feature)
            clean_features_list.append(clean_series)
            
        clean_features_df = pd.concat([feature for feature in clean_features_list], axis=1)
        
        # ======= III. Recreate the DataFrame =======
        non_features_df = new_df[self.non_feature_columns].copy()
        final_df = pd.concat([non_features_df, clean_features_df], axis=1)
        final_df = final_df.ffill()
        final_df = final_df.dropna(axis=0)
        
        return final_df
        


#! ==================================================================================== #
#! =================================== Sampling  ====================================== #
class DataSampler():
    def __init__(
        self, 
        data: pd.DataFrame,
        n_jobs: int = 1
    ):
        # ======= II. Store the inputs =======
        self.data = data
        self.n_jobs = n_jobs
        
        self.available_methods = {
            "daily_volBars": sampl.daily_volBars,
            "daily_cumsumTargetBars": sampl.daily_cumsumTargetBars,
            "daily_cumsumWeightedTargetBars": sampl.daily_cumsumWeightedTargetBars,
        }
        
        # ======= III. Initialize Results =======
        self.params = None
        self.resampled_data = None
    
    #?__________________________________________________________________________________ #
    def set_params(
        self,
        sampling_method: str = "daily_volBars",
        column_name: str = "close",
        grouping_column: str = "date",
        new_cols_methods: str = "mean",
        target_bars: int = 100,
        window_bars_estimation: int = 10,
        pre_threshold: float = 1000,
        weight_column_name: str = "close",
        vol_threshold: float = 0.0005,
        aggregation_dict: dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "ts": ["first", "last"],
            "date": "first",
            "bid_open": "first",
            "ask_open": "first",
        },
    ):
        self.params = {
            'sampling_method': sampling_method,
            'column_name': column_name,
            'grouping_column': grouping_column,
            'new_cols_methods': new_cols_methods,
            'target_bars': target_bars,
            'window_bars_estimation': window_bars_estimation,
            'pre_threshold': pre_threshold,
            'weight_column_name': weight_column_name,
            'vol_threshold': vol_threshold,
            'aggregation_dict': aggregation_dict,
        }
        
        return self

    #?____________________________________________________________________________________ #
    def extract(self):
        
        def filter_params_for_function(func, param_dict):
            sig = inspect.signature(func)
            valid_keys = sig.parameters.keys()
            
            return {k: v for k, v in param_dict.items() if k in valid_keys}
        
        # ======= I. Check Sampling Method =======
        if self.params['sampling_method'] not in self.available_methods.keys():
            raise ValueError(f"Sampling method {self.params['sampling_method']} is not available.")
        
        sampling_method = self.available_methods[self.params['sampling_method']]
        
        # ======= II. Apply Sampling =======
        resampled_data = sampling_method(data=self.data, **filter_params_for_function(sampling_method, self.params))
        
        # ======= III. Store Results =======
        self.resampled_data = resampled_data
        
        return resampled_data
