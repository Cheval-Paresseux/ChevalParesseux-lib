from ..Prepping import data_verification as verif

import pandas as pd
import numpy as np
from typing import Union
from joblib import Parallel, delayed


#! ==================================================================================== #
#! ==================================== Preparator ==================================== #
class TrainingPreparator():
    def __init__(
        self,
        training_data: Union[list, pd.DataFrame],
    ):
        # ======= I. Store the training data =======
        self.training_data = training_data
        
        # ======= II. Set default parameters =======
        self.stationarity_threshold = None
        self.outliers_threshold = None
        self.mean_tolerance = None
        self.std_tolerance = None
        self.range_tolerance = None
        self.n_jobs = 1
        
        # ======= III. Post Processing Information =======
        self.processed_data = None
        self.features_informations = None
        
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        stationarity_threshold: float = 0.05, 
        outliers_threshold: float = 3, 
        mean_tolerance: float = 0.01, 
        std_tolerance: float = 0.01, 
        range_tolerance: float = 0.1,
        n_jobs: int = 1
    ):
        """
        This function sets the parameters for the training preparator.
        Parameters:
            - stationarity_threshold (float): The threshold for stationarity tests.
            - outliers_threshold (float): The threshold for outlier detection.
            - mean_tolerance (float): The tolerance for the mean value.
            - std_tolerance (float): The tolerance for the standard deviation.
            - range_tolerance (float): The tolerance for the range of values.
        """
        self.stationarity_threshold = stationarity_threshold
        self.outliers_threshold = outliers_threshold
        self.mean_tolerance = mean_tolerance
        self.std_tolerance = std_tolerance
        self.range_tolerance = range_tolerance
        self.n_jobs = n_jobs
    
    #?____________________________________________________________________________________ #
    def check_feature(
        self, 
        feature_series: pd.Series,
        stationarity_threshold: float = 0.05, 
        outliers_threshold: float = 3, 
        mean_tolerance: float = 0.01, 
        std_tolerance: float = 0.01, 
        range_tolerance: float = 0.1,
    ):
        """
        This function checks a given feature series for various characteristics and store the results in a DataFrame.
        Those values should be used to determine if the feature is suitable for training.
        If some transformations are applied, you should use the transformation values for the testing data.
        Parameters:
            - feature_series (pd.Series): The feature series to be checked.
            - stationarity_threshold (float): The threshold for stationarity tests.
            - outliers_threshold (float): The threshold for outlier detection.
            - mean_tolerance (float): The tolerance for the mean value.
            - std_tolerance (float): The tolerance for the standard deviation.
            - range_tolerance (float): The tolerance for the range of values.
        
        Returns:
            - scaled_series (pd.Series): The normalized series.
            - results_df (pd.DataFrame): A DataFrame containing the results of the checks.
        """
        # ======= I. Check for error values =======
        clean_series, error_proportion, beginning_nans, middle_nans, infinite_indexes = verif.check_for_error_values(
            feature_series=feature_series
        )
        
        # ======= II. Check for outliers =======
        filtered_series, outliers_df = verif.check_for_outliers(feature_series=clean_series,  threshold=outliers_threshold)
        
        # ======= III. Check for scale =======
        scaled_series, mean, std, min_val, max_val = verif.check_for_scale(
            feature_series=filtered_series, 
            mean_tolerance=mean_tolerance, 
            std_tolerance=std_tolerance, 
            range_tolerance=range_tolerance
        )
        
        # ======= IV. Check for stationarity =======
        dropped_series = scaled_series.dropna()
        is_adf_stationary, is_kpss_stationary = verif.check_for_stationarity(feature_series=dropped_series, threshold=stationarity_threshold)

        # ======= V. Store results inside a DataFrame =======
        results_df = pd.DataFrame({
            "feature_name": feature_series.name,
            "error_proportion": error_proportion,
            "beginning_nans": len(beginning_nans),
            "middle_nans": len(middle_nans),
            "infinite_indexes": len(infinite_indexes),
            "outliers_count": len(outliers_df),
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "is_adf_stationary": is_adf_stationary,
            "is_kpss_stationary": is_kpss_stationary
        }, index=[0])
        
        return scaled_series, results_df
    
    #?____________________________________________________________________________________ #
    def check_all_features(
        self,
        data_df: pd.DataFrame,
    ):
        # ======= I. Check all features =======
        features_df = data_df.copy()

        scaled_series, infos_dfs = Parallel(n_jobs=self.n_jobs)(
            delayed(self.check_feature)(
                feature_series=features_df[feature], 
                stationarity_threshold=self.stationarity_threshold, 
                outliers_threshold=self.outliers_threshold, 
                mean_tolerance=self.mean_tolerance, 
                std_tolerance=self.std_tolerance, 
                range_tolerance=self.range_tolerance
            ) for feature in features_df.columns
        )
        
        # ======= II. Recreate the Feature DataFrame =======
        scaled_data = pd.DataFrame(scaled_series, columns=features_df.columns)
        
        # ======= III. Store the informations =======
        features_informations = pd.concat(infos_dfs, ignore_index=True)
        
        return scaled_data, features_informations
    
    
    
    

#*____________________________________________________________________________________ #