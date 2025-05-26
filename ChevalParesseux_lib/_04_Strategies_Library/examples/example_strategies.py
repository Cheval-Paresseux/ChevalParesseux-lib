from ..._04_Strategies_Library import common as com
from ... import _01_Data_Processing as proc
from ... import _02_Signal_Extraction as extrac
from ... import _03_Strategy_Tuning as tune


import pandas as pd
import numpy as np
from typing import Union, Self
from joblib import Parallel, delayed



class Simple_momentum(com.Strategy):
    """
    Simple Momentum Strategy.
    
    This strategy uses a combination of features based on moving averages, momentum, and volatility to generate trading signals.
    It includes a labeller for defining the target variable, a feature selector for reducing dimensionality, and a predictor for generating signals.
    The strategy is designed to be flexible and can be tuned with various parameters for each component.
    
    The strategy aims to predict the directio of a single asset's price movement based on historical data.
    It outputs a DataFrame containing the signals, sizes, and other relevant columns for trading operations.
    """
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(
        self, 
        n_jobs: int = 1,
    ) -> None:
        """
        Constructor for the Simple_momentum strategy.
        
        Parameters:
            - n_jobs (int): The number of jobs to run in parallel, default is 1.
        """
        # ======= I. Jobs =======
        self.n_jobs = n_jobs
        self.params = {}

        # ======= II. Models =======
        self.selector = None
        self.predictor = None

    #?____________________________________________________________________________________ #
    def set_params(
        self,
        non_features: list = None,
        labeller_params: dict = None,
        average_feature_params: dict = None,
        momentum_feature_params: dict = None,
        vol_feature_params: dict = None,
        obs_selector_params: dict = None,
        selector_params: dict = None,
        tuner_params: dict = None,
        signal_processor_params: dict = None,
        sizer_params: dict = None,
    ):
        """
        Set the parameters for the strategy.

        Parameters:
            - non_features (list): List of columns that are not features.
            - labeller_params (dict): Parameters for the labeller.
            - average_feature_params (dict): Parameters for the average feature.
            - momentum_feature_params (dict): Parameters for the momentum feature.
            - vol_feature_params (dict): Parameters for the volatility feature.
            - obs_selector_params (dict): Parameters for the observation selector.
            - selector_params (dict): Parameters for the selector.
            - tuner_params (dict): Parameters for the tuner.
            - signal_processor_params (dict): Parameters for the signal processor.
            - sizer_params (dict): Parameters for the sizer.
        """
        # ======= I. Set parameters =======
        self.non_features = non_features
        self.params = {
            "labeller_params": labeller_params,
            "average_feature_params": average_feature_params,
            "momentum_feature_params": momentum_feature_params,
            "vol_feature_params": vol_feature_params,
            "obs_selector_params": obs_selector_params,
            "selector_params": selector_params,
            "tuner_params": tuner_params,
            "signal_processor_params": signal_processor_params,
            "sizer_params": sizer_params,
        }

        return self

    #?________________________________ Auxiliary methods _________________________________ #
    def get_features_labels(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extracts features and labels from the input data.
        
        This method applies a series of transformations to the input data to generate features based on moving averages, momentum, and volatility.
        It also labels the data using a triple barrier method to define the target variable for the strategy.
        
        Parameters:
            - data (pd.DataFrame): DataFrame containing the input data with a 'close' column.
        
        Returns:
            - processed_data (pd.DataFrame): DataFrame containing the original data along with the extracted features and labels.
        """
        # ======= 0. Copy data =======
        processed_data = data.copy()
        
        # ======= I. labelling =======
        labeller = proc.TripleBarrier_labeller()
        labeller_params = self.params['labeller_params']
        labeller.set_params(**labeller_params)

        labels_df = labeller.extract(data=processed_data['close'])
        processed_data['label'] = labels_df[labels_df.columns[0]]
        
        # ======= II. Moving Average Based Feature =========
        avg_feature = proc.Average_feature(n_jobs=self.n_jobs)
        average_feature_params = self.params['average_feature_params']
        avg_feature.set_params(**average_feature_params)

        average_feature_df = avg_feature.extract(data=processed_data['close'])
        processed_data = pd.concat([processed_data, average_feature_df], axis=1)
        
        # ======= III. Momentum Based Feature =========
        momentum_feature = proc.Momentum_feature(n_jobs=self.n_jobs)
        momentum_feature_params = self.params['momentum_feature_params']
        momentum_feature.set_params(**momentum_feature_params)

        momentum_feature_df = momentum_feature.extract(data=processed_data['close'])
        processed_data = pd.concat([processed_data, momentum_feature_df], axis=1)

        # ======= IV. Volatility Based Feature =========
        vol_feature = proc.Volatility_feature(n_jobs=self.n_jobs)
        vol_feature_params = self.params['vol_feature_params']
        vol_feature.set_params(**vol_feature_params)

        vol_feature_df = vol_feature.extract(data=processed_data['close'])
        processed_data = pd.concat([processed_data, vol_feature_df], axis=1)
        
        return processed_data
    
    #?____________________________________________________________________________________ #
    def process_data(
        self,
        data: pd.DataFrame,
    ) -> tuple:
        """
        Processes the input data to extract features and labels, and applies feature selection.
        
        This method is designed to be called before fitting the model or making predictions.
        It extracts features and labels from the input data, applies a feature selection model, and prepares the data for training or prediction.
        
        Parameters:
            - data (pd.DataFrame): DataFrame containing the input data with a 'close' column.
        
        Returns:
            - processed_data (pd.DataFrame): DataFrame containing the original data along with the extracted features and labels.
            - X (pd.DataFrame): DataFrame containing the features used for training or prediction.
            - y (pd.Series): Series containing the labels corresponding to the features.
        """
        # ======= I. Extract Features and Labels =======
        processed_data = self.get_features_labels(data=data).copy()
        
        # ======= II. Feature Selection Model =========
        processed_data = self.selector.extract(data=processed_data)
        
        # ======= III. Extracting the features =======
        processed_data = processed_data.dropna(axis=0, how='any').copy()
        X = processed_data.drop(columns=self.non_features).copy()
        y = processed_data['label'].copy()
        
        return processed_data, X, y
    
    #?____________________________________________________________________________________ #
    def fit(
        self,
        data: pd.DataFrame,
    ) -> Self:
        """
        Fits the Simple_momentum strategy to the input data.
        
        This method processes the input data to extract features and labels, applies resampling, feature selection, and predictor tuning.
        It prepares the strategy for making predictions on new data.
        
        Parameters:
            - data (pd.DataFrame): DataFrame containing the input data with a 'close' column.
        
        Returns:
            - self (Simple_momentum): The fitted strategy instance.
        """
        # ======= 0. Prepare data =======
        processed_data = self.get_features_labels(data=data).copy()
        
        # ======= I. Resampling =========
        obs_selector = proc.Temporal_uniqueness_selection(n_jobs=self.n_jobs, random_state=72)
        obs_selector_params = self.params['obs_selector_params']
        obs_selector.set_params(**obs_selector_params)
        
        datasets = obs_selector.extract(data=processed_data)
        resampled_df = datasets[0][0]
        
        resampled_df = resampled_df.sort_values(by='date').copy()
        resampled_df.reset_index(drop=True, inplace=True)
        
        # ======= II. Features Selection =======
        selector = extrac.Correlation_selector(n_jobs=self.n_jobs)
        selector_params = self.params['selector_params']
        selector.set_params(**selector_params)

        features_df = resampled_df.drop(columns=self.non_features).copy()
        selector.fit(data=features_df)

        training_df = selector.extract(data=resampled_df)
        training_df = training_df.sort_values(by='date').copy()
        training_df.reset_index(drop=True, inplace=True)
        self.selector = selector
        
        # ======= III. Predictor Tuning =======
        X_train = training_df.drop(columns=self.non_features).copy()
        y_train = training_df['label'].copy()
        
        tuner = extrac.Classifier_gridSearch(n_jobs=self.n_jobs, random_state=72)
        tuner_params = self.params['tuner_params']['tuning_params']
        tuner.set_params(**tuner_params)

        model = extrac.SKL_randomForest_classifier
        grid_universe = self.params['tuner_params']['grid_universe']
        criteria = self.params['tuner_params']['criteria']

        nb_observations = len(training_df)
        n_folds = 3
        size_fold = nb_observations // n_folds
        data_folds = []
        for i in range(0, n_folds):
            start_idx = i * size_fold
            end_idx = (i + 1) * size_fold
            X_fold = training_df.iloc[start_idx : end_idx].drop(columns=self.non_features).copy()
            y_fold = training_df.iloc[start_idx : end_idx]['label'].copy()
            data_folds.append((X_fold, y_fold))

        tuner.fit(model=model, grid_universe=grid_universe, data=data_folds, criteria=criteria)
        
        data_train = (X_train, y_train)
        fitted_model = tuner.extract(model=model, data=data_train)
        self.predictor = fitted_model
        
        return self
    
    #?__________________________________ Prediction methods ______________________________ #
    def extract(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Makes predictions on the data using the trained model.
        
        This method processes the input data to extract features, applies the trained predictor to generate signals,
        and processes the signals to determine the size of positions based on volatility.
        
        Parameters:
            - data (pd.DataFrame): DataFrame containing the data to predict.
        
        Returns:
            - signals_df (pd.DataFrame): DataFrame containing the predictions.
        """
        # ======= I. Process data =======
        processed_data, X, y = self.process_data(data=data)
        signals_df = processed_data.copy()
        
        # ======= II. Predict =======
        signals = self.predictor.predict(X)
        
        # ======= III. Signal Processing =======
        signal_processor = tune.Confirmation_processor(n_jobs=self.n_jobs)
        signal_processor_params = self.params['signal_processor_params']
        signal_processor.set_params(**signal_processor_params)

        signals_df['signal'] = signal_processor.extract(signal_series=signals)
        
        # ======= III. Size Positions =======
        sizer = tune.Volatility_sizer(n_jobs=self.n_jobs)
        sizer_params = self.params['sizer_params']
        sizer.set_params(**sizer_params)
        
        signals_df['size'] = sizer.extract(data=signals_df)
        signals_df['size'] = signals_df['size'].fillna(0)
        
        return signals_df

