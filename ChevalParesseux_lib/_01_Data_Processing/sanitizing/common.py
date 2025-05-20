from ..sanitizing import sanitizing_tools as sanit

import pandas as pd
import numpy as np
from typing import Union, Self
from joblib import Parallel, delayed
import inspect

#! check_error_features has some strange scale filter

#! ==================================================================================== #
#! ==================================== Cleaning ====================================== #
class Features_sanitizer():
    """
    Cleaner class for preprocessing time series features data.
    
    This class is designed to handle the following tasks:
        - Handling error features (e.g., NaN, infinite, constant columns)
        - Checking the features for marginal errors, outliers, scale and stationarity
        - Extracting rules for feature processing on test data
        - Extracting clean features from new data based on the rules defined during training
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        # ======= I. Store the inputs =======
        self.n_jobs = n_jobs

        # ======= II. Initialize Results =======
        self.params = {}
        self.features_informations = None
        self.features_rules = None
        self.error_features = []
    
    #?__________________________________________________________________________________ #
    def set_params(
        self,
        stationarity_threshold: float = 0.05,
        outliers_threshold: float = 5,
    ) -> Self:
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
    def extract(
        self,
        training_data: Union[list, pd.DataFrame],
        non_feature_columns: list = ['open', 'high', 'low', 'close', 'volume'],
    ) -> list:
        """
        Cleans the training data by performing the following steps:
        
            1. Check for error features (e.g., NaN, infinite, constant columns)
            2. Check the features for marginal errors, outliers, scale and stationarity
            3. Vertical stacking of DataFrames
            4. Extracting rules for feature processing on test data
        
        Parameters:
            - training_data (Union[list, pd.DataFrame]): The training data to be processed.
            - non_feature_columns (list): List of columns that are not features.
        
        Returns:
            - processed_data (list): List of processed DataFrames.
        """
        # ======= I. Verify the input type =======
        if not isinstance(training_data, list):
            train_data = [training_data]
        else:
            train_data = training_data.copy()
            
        # ======= II Check for Error Features =======
        error_features = self.check_error_features(df_list=train_data)
        self.error_features = error_features
        
        # ======= III. Check the Features DataFrames =======
        features_informations = []
        processed_data = []
        idx = 0
        for individual_df in train_data:
            tested_df = individual_df.drop(columns=non_feature_columns, axis=1).copy()
            
            # III.1 Drop the error features
            if len(self.error_features) > 0:
                tested_df = tested_df.drop(columns=self.error_features, axis=1)

            # III.2 Check the features
            scaled_features, features_infos = self.check_features_df(training_df=tested_df)
            
            # III.3 Rearrange the final DataFrame
            non_feature_df = individual_df[non_feature_columns].copy()
            clean_df = pd.concat([non_feature_df, scaled_features], axis=1)
            clean_df = clean_df.ffill()
            clean_df = clean_df.dropna(axis=0)
            
            # III.4 Add the df index to the features information
            features_infos['df_index'] = idx
            idx += 1
            
            processed_data.append(clean_df)
            features_informations.append(features_infos)

        # ======= II. Extract Informations =======
        features_informations = pd.concat(features_informations, axis=0, ignore_index=True)
        features_rules = self.get_rules(features_informations)
        
        # ======= III. Store Results =======
        self.features_informations = features_informations
        self.features_rules = features_rules
        
        return processed_data

    #?____________________________________________________________________________________ #
    def extract_new(
        self, 
        new_data: pd.DataFrame,
        non_feature_columns: list = ['open', 'high', 'low', 'close', 'volume'],
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
        features_list = [column for column in columns_name if column not in non_feature_columns]
        features_list = [feature for feature in features_list if feature not in self.error_features]
        
        features_df = new_df[features_list].copy()
        
        # ======= II. Extract the clean features =======
        clean_features_list = []
        for feature in features_df.columns:
            clean_series = check_feature(feature_series=features_df[feature], feature_name=feature)
            clean_features_list.append(clean_series)
            
        clean_features_df = pd.concat([feature for feature in clean_features_list], axis=1)
        
        # ======= III. Recreate the DataFrame =======
        non_features_df = new_df[non_feature_columns].copy()
        final_df = pd.concat([non_features_df, clean_features_df], axis=1)
        final_df = final_df.ffill()
        final_df = final_df.dropna(axis=0)
        
        return final_df
        

