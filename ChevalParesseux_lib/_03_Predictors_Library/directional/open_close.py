from ... import _01_Data_Processing as dlib
from ... import _02_Predictive_Models as plib
from ... import utils
from ..directional import common as com

import pandas as pd
import numpy as np
from typing import Optional, Self
from tqdm import tqdm



class OtC_directional(com.Directional_Model):
    """
    Directional Model for Open-to-Close (OtC) trading strategy.
    
    This model is designed to extract features from OHLC daily data, then a signal is generated and should be used to trade the same day.
    Informations used for signal[t] are the ones available until open[t], a trade should be executed at open[t + 1minutes].
    """
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Constructor for the OtC_directional model.
        
        Parameters:
            - n_jobs (int): Number of jobs to run in parallel. Default is 1.
        """
        # ======= I. Initialize Class =======
        # super().__init__(n_jobs=n_jobs)
        self.n_jobs = n_jobs
        
        # ======= II. Initialize Auxilaries =======
        self.cols_order = None
    
    #?____________________________________________________________________________________ #
    def get_default_params(
        self,
    ) -> dict:
        """
        Gets the default parameters for the OtC directional model.
        
        Returns:
            - dict: A dictionary containing the default parameters for the model.
        """
        default_params = {}
        
        # ======= I. Features Models =======
        # --- 1. Parameters Template ---
        general_params = {
            "window": [5, 10, 20, 30],
            "smoothing_method": [None, "ewma"],
            "window_smooth": [5, 10, 20, 30],
            "lambda_smooth": [0.2],
        }
        
        nosmooth_params = {
            **general_params,
            "smoothing_method": [None],
            "window_smooth": [5, 10],
        }

        quantile_params = {
            **general_params,
            "quantile": [0.05, 0.25, 0.5, 0.75, 0.95],
            "window_smooth": [5, 10],
        }

        kama_params = {
            "window": [20, 30],
            "fastest_window": [5, 10],
            "slowest_window": [15, 20],
            "smoothing_method": [None, "ewma"],
            "window_smooth": [5, 10],
            "lambda_smooth": [0.2],
        }
        
        # --- 2. Features Model Registry ---
        # i. Features using general params
        features_general = {
            "average": dlib.Average_feature,
            "median": dlib.Median_feature,
            "minimum": dlib.Minimum_feature,
            "maximum": dlib.Maximum_feature,
            "shannon": dlib.Shannon_entropy_feature,
            "plugin": dlib.Plugin_entropy_feature,
            "lempelZiv": dlib.LempelZiv_entropy_feature,
            "kontoyiannis": dlib.Kontoyiannis_entropy_feature,
            "momentum": dlib.Momentum_feature,
            "linear_tempreg": dlib.Linear_tempReg_feature,
            "non_linear_tempreg": dlib.Nonlinear_tempReg_feature,
            "stochasticRSI": dlib.StochasticRSI_feature,
            "rsi": dlib.RSI_feature,
            "ehlersFisher": dlib.EhlersFisher_feature,
            "oscillator": dlib.Oscillator_feature,
            "vortex": dlib.Vortex_feature,
            "vigor": dlib.Vigor_feature,
            "stochasticOscillator": dlib.StochasticOscillator_feature,
        }

        # ii. Features using no-smoothing params 
        features_nosmooth = {
            "volatility": dlib.Volatility_feature,
            "skewness": dlib.Skewness_feature,
            "kurtosis": dlib.Kurtosis_feature,
            "Z_momentum": dlib.Z_momentum_feature,
        }

        # iii. Features with custom params 
        features_custom = {
            "quantile": (dlib.Quantile_feature, quantile_params),
            "kama": (dlib.Kama_feature, kama_params),
        }

        # --- 3. Final Features Model Dictionary ---
        features_models = {
            **{k: (v, general_params) for k, v in features_general.items()},
            **{k: (v, nosmooth_params) for k, v in features_nosmooth.items()},
            **features_custom,
        }
        default_params['features_models'] = features_models
        
        # ======= II. Labeller Model =======
        labeller_params = {
            "threshold": [0.3],
            "vol_window": [21],
            "smoothing_method": [None],
            "window_smooth": [0],
            "lambda_smooth": [0],
        }
        labeller_models = (dlib.Naive_labeller, labeller_params)
        default_params['labeller_models'] = labeller_models
        
        # ======= III. Resampler Model =======
        resampler_params = {
            'label_column': ['label'],
            'price_column': ['intra_close'],
            'n_samples': [1.3],
            'replacement': [True],
            'balancing': [True],
            'vol_window': [21],
            'upper_barrier': [0.3],
            'vertical_barrier': [21],
            'grouping_column': [None],
        }
        resampler_models = (dlib.Temporal_uniqueness_selection, resampler_params)
        default_params['resampler_models'] = resampler_models
        
        # ======= IV. Feature Selector Model =======
        selector_params = {
            'correlation_threshold': [0.9],
        }
        selector_models = (plib.Correlation_selector, selector_params)
        default_params['selector_models'] = selector_models
        
        # ======= V. Tuner Model =======
        tuner_params = {
            'random_search': True,
            'n_samples': 500,
        }
        tuner_models = (plib.Classifier_gridSearch, tuner_params)
        default_params['tuner_models'] = tuner_models
        
        # ======= VI. Predictor Model =======
        predictor_models = plib.SKL_randomForest_classifier
        default_params['predictor_models'] = predictor_models
        
        # ======= VII. Grid Universe =======
        grid_universe = {
            'raw_predict': ['False'],  # whether to use raw predictions or not
            'min_proba': [0.7],  # minimum probability threshold for predictions
            'n_estimators': [100],  # number of trees in the forest
            'criterion': ['gini', 'entropy', 'log_loss'],  # log_loss only for classifier in sklearn >= 1.1
            'max_depth': [None, 3, 4, 5, 6, 10, 20],  # allow unlimited depth with None
            'min_samples_split': [2, 5, 10, 20, 50, 100],  # smaller values allow for deeper trees
            'min_samples_leaf': [1, 2, 5, 10, 20, 50, 100],  # controls leaf size
            'max_features': [None, 'sqrt', 'log2', 0.5, 0.7], 
            'bootstrap': [True, False],  # whether bootstrap samples are used
            'class_weight': [None, 'balanced', 'balanced_subsample'],  # useful for imbalanced classes
        }
        default_params['grid_universe'] = grid_universe
        
        # ======= VIII. Grid Criteria =======
        grid_criteria = 'accuracy'
        default_params['grid_criteria'] = grid_criteria
        
        return default_params
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        #_____ features models _____ #
        features_models: Optional[dict] = None,
        #_____ labeller models _____ #
        labeller_models: Optional[tuple] = None,
        #_____ training models _____ #
        resampler_models: Optional[tuple] = None,
        selector_models: Optional[tuple] = None,
        tuner_models: Optional[tuple] = None,
        #_____ fitting parameters _____ #
        predictor_models: Optional[object] = None,
        grid_universe: Optional[dict] = None,
        grid_criteria: Optional[str] = None,
    ) -> Self:
        """
        Sets the parameters for the OtC directional model.
        
        Parameters:
            - features_models (Optional[dict]): Dictionary of feature extraction models and their parameters.
            - labeller_models (Optional[tuple]): Tuple containing the labeller model and its parameters.
            - resampler_models (Optional[tuple]): Tuple containing the resampler model and its parameters.
            - selector_models (Optional[tuple]): Tuple containing the selector model and its parameters.
            - tuner_models (Optional[tuple]): Tuple containing the tuner model and its parameters.
            - predictor_models (Optional[object]): Tuple containing the predictor model.
            - grid_universe (Optional[dict]): Dictionary defining the grid universe for hyperparameter tuning.
            - grid_criteria (Optional[str]): Criteria for selecting the best hyperparameters.
        """
        # ======= I. Extract Default Parameters =======
        default_params = self.get_default_params()
        
        # ======= II. Set Parameters =======
        self.params = {
            'features_models': features_models if features_models is not None else default_params['features_models'],
            'labeller_models': labeller_models if labeller_models is not None else default_params['labeller_models'],
            'resampler_models': resampler_models if resampler_models is not None else default_params['resampler_models'],
            'selector_models': selector_models if selector_models is not None else default_params['selector_models'],
            'tuner_models': tuner_models if tuner_models is not None else default_params['tuner_models'],
            'predictor_models': predictor_models if predictor_models is not None else default_params['predictor_models'],
            'grid_universe': grid_universe if grid_universe is not None else default_params['grid_universe'],
            'grid_criteria': grid_criteria if grid_criteria is not None else default_params['grid_criteria'],
        }
        
        return self
    
    #?_______________________________ Processing methods _________________________________ #
    def process_decomposition(
        self,
        data: pd.DataFrame
    ) -> dict:
        """
        Performs series decomposition on the input DataFrame.
        
        Parameters:
            - data (pd.DataFrame): The input DataFrame containing financial data with columns ['date', 'open', 'close', 'high', 'low'].
        
        Returns:
            - dict: A dictionary containing decomposed series DataFrames for 'intraday', 'overnight', and 'complete'.
        """
        # ======= 0. Validate Input DataFrame =======
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        required_columns = ['date', 'open', 'close', 'high', 'low']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # ======= I. Decompose Series =======
        intraday_df, overnight_df, complete_df, daily_data = utils.get_series_decomposition(data=data)
        series_dict = {"intraday": intraday_df, "overnight": overnight_df, "complete": complete_df, "daily_data": daily_data}
        
        return series_dict
        
    #?____________________________________________________________________________________ #
    def process_features(
        self, 
        series_dict: dict
    ) -> pd.DataFrame:
        """
        Extracts features from the given series data.
        
        Parameters:
            - series_dict (dict): A dictionary containing series DataFrames for 'intraday', 'overnight', and 'complete'.
        
        Returns:
            - pd.DataFrame: A DataFrame containing the extracted features for each series.
        """        
        # ======= I. Feature Extraction =======
        # --- 1. Initialize features models ---
        features_model = self.params['features_models']
        features_dfs = []

        special_inputs = {
            "ehlersFisher": lambda o, h, l, c: (h, l),
            "vortex": lambda o, h, l, c: (c, h, l),
            "vigor": lambda o, h, l, c: (o, c, h, l),
            "stochasticOscillator": lambda o, h, l, c: (c, h, l),
        }
        
        # --- 2. Extract features useful series ---
        series = {key: value for key, value in series_dict.items() if key in ['intraday', 'overnight', 'complete']}

        # --- 3. Extract features for each series ---
        print("Extracting features...")
        for prefix, series_df in tqdm(series.items()):
            o, h, l, c = (series_df['open'], series_df['high'], series_df['low'], series_df['close'])

            series_features = []
            for feature_name, (feature_model, feature_params) in features_model.items():
                model = feature_model(n_jobs=self.n_jobs)
                model.set_params(**feature_params)

                input_data = special_inputs.get(feature_name, lambda o, h, l, c: c)(o, h, l, c)
                feature_df = model.extract(data=input_data)
                feature_df.columns = [f"{prefix}_{col}" for col in feature_df.columns]

                series_features.append(feature_df)

            # Combine features for one series (intraday/overnight/complete)
            series_features_df = pd.concat(series_features, axis=1)
            features_dfs.append(series_features_df)

        # ======= II. Concatenate all features =======
        features_df = pd.concat(features_dfs, axis=1)
        
        return features_df
    
    #?____________________________________________________________________________________ #
    def process_labels(
        self,
        intraday_df: pd.DataFrame
    ) -> pd.Series:
        """
        Extracts labels from the intraday DataFrame using the labeller model.
        
        Parameters:
            - intraday_df (pd.DataFrame): The DataFrame containing intraday data with a 'close' column.
            
        Returns:
            - pd.Series: A Series containing the extracted labels, renamed to 'label'.
        """
        # ======= I. Set up labeller =======
        labeller_model, labeller_params = self.params['labeller_models']
        labeller = labeller_model(n_jobs=self.n_jobs)
        labeller.set_params(**labeller_params)
        
        # ======= II. Extract labels =======
        labels = labeller.extract(data=intraday_df['close'])
        labels_series = labels[labels.columns[0]].copy()
        labels_series = labels_series.rename('label')

        return labels_series
    
    #?____________________________________________________________________________________ #
    def process_train_data(
        self, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes the training data for the OtC directional model.
        
        Parameters:
            - data (pd.DataFrame): The input DataFrame containing financial data with columns ['date', 'open', 'close', 'high', 'low'].
        
        Returns:
            - pd.DataFrame: A DataFrame containing the processed features and labels for training.
        """
        # ======= I. Pre-process data =======
        data_df = data.copy()
        data_df = data_df[['date', 'open', 'close', 'high', 'low']]
        
        # ======= II. Process Features & Labels =======
        series_dict = self.process_decomposition(data=data_df)
        
        features_df = self.process_features(series_dict=series_dict)
        labels_series = self.process_labels(intraday_df=series_dict['intraday'])
        
        # ======= III. Combine Features and Labels into a single DataFrame =======
        intra_close = series_dict['intraday']['close']
        features_df['intra_close'] = intra_close
        
        processed_data = pd.concat([features_df, labels_series], axis=1)
        processed_data['label'] = processed_data['label'].shift(-1) # to align labels with the next row
        processed_data.reset_index(inplace=True)
        processed_data.rename(columns={'index': 'date'}, inplace=True)

        return processed_data
    
    #?________________________________ Fitting methods ___________________________________ #
    def fit(
        self,
        processed_train_data: pd.DataFrame,
    ) -> tuple:
        """
        Fits the OtC directional model using the provided training data.
        
        Parameters:
            - processed_train_data (pd.DataFrame): The pre-processed training data containing features and labels.
        
        Returns:
            - tuple: A tuple containing the training features (X_train) and labels (y_train).
        """
        # ======= 0. Set non-features (always the same if transformed data from transform_data method) =======
        non_features = ['date', 'label', 'intra_close']
        train_data = processed_train_data.dropna(axis=0).copy()
        
        # ======= I. Resample Training Data =======
        resampler_model, resampler_params = self.params['resampler_models']
        resampler = resampler_model(n_jobs=self.n_jobs, random_state=72)
        resampler.set_params(**resampler_params)
        
        datasets = resampler.extract(data=train_data)
        resampled_df = datasets[0][0]
        resampled_df.sort_values(by='date', inplace=True)
        resampled_df.reset_index(drop=True, inplace=True)
        
        # ======= II. Feature Selection =======
        features_df = resampled_df.drop(columns=non_features).copy()

        selector_model, selector_params = self.params['selector_models']
        selector = selector_model(n_jobs=self.n_jobs)
        selector.set_params(**selector_params)

        selector.fit(data=features_df)
        self.selector = selector
        
        train_df = selector.extract(data=resampled_df)

        # ======= III. Fit the predictor =======
        # ----- 1. Tuner Model -----
        tuner_model, tuner_params = self.params['tuner_models']
        tuner = tuner_model(n_jobs=self.n_jobs, random_state=72)
        tuner.set_params(**tuner_params)

        # ----- 2. Fitting Model -----
        model = self.params['predictor_models']
        grid_universe = self.params['grid_universe']
        criteria = self.params['grid_criteria']

        nb_observations = len(train_df)
        n_folds = 3
        size_fold = nb_observations // n_folds
        data = []
        for i in range(0, n_folds):
            start_idx = i * size_fold
            end_idx = (i + 1) * size_fold
            X_fold = train_df.iloc[start_idx : end_idx].drop(columns=non_features).copy()
            y_fold = train_df.iloc[start_idx : end_idx]['label'].copy()
            data.append((X_fold, y_fold))

        tuner.fit(model=model, grid_universe=grid_universe, data=data, criteria=criteria)
        best_params = tuner.best_params
        print(f'With a {criteria} of {tuner.best_score:.2f}, Best parameters : {best_params}')

        X_train = train_df.drop(columns=non_features).copy()
        y_train = train_df['label'].copy()
        data_train = (X_train, y_train)
        
        fitted_model = tuner.extract(model=model, data=data_train)
        self.predictor = fitted_model
        
        self.cols_order = X_train.columns.tolist()  # Store the order of columns
        
        return X_train, y_train

    #?________________________________ Predict methods ___________________________________ #
    def process_data(
        self,
        data: pd.DataFrame,
    ) -> list:
        """
        Transforms the input data into a format suitable for prediction.
        
        Parameters:
            - data (pd.DataFrame): The input DataFrame containing financial data with columns ['date', 'open', 'close', 'high', 'low'].
        
        Returns:
            - list: A list of DataFrames, each containing the processed features and labels for a single day.
        """
        # ======= I. Pre-process data =======
        data_df = data.copy()
        data_df = data_df[['date', 'open', 'close', 'high', 'low']]
        
        # ======= II. Process Features & Labels =======
        series_dict = self.process_decomposition(data=data_df)
        
        features_df = self.process_features(series_dict=series_dict)
        labels_series = self.process_labels(intraday_df=series_dict['intraday'])
        
        intra_close = series_dict['intraday']['close']
        features_df['intra_close'] = intra_close
        
        # ======= V. Combine Features and Labels into a single DataFrame =======
        big_df = pd.concat([features_df, labels_series], axis=1)
        big_df['label'] = big_df['label'].shift(-1) # to align labels with the day we actually trade it 
        big_df = big_df.shift(1)  # Shift to align with the actual operation day (as we then merge day by day it is necessary because we won't shift the signals)
        
        # ======= VI. Merge daily data with big_df =======
        big_df = self.selector.extract(data=big_df)
        big_df.reset_index(inplace=True)
        big_df.rename(columns={'index': 'date'}, inplace=True)
        
        # Convert daily_data list into a single DataFrame
        daily_data = series_dict['daily_data']
        combined_daily_df = pd.concat(daily_data, ignore_index=True)

        # Merge with feature data on date
        enriched_df = combined_daily_df.merge(big_df, on='date', how='left')
        enriched_df = enriched_df.dropna(axis=0)
        daily_data = [group for _, group in enriched_df.groupby('date')]

        return daily_data

    #?____________________________________________________________________________________ #
    def get_signals(
        self,
        day_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # ======= I. Prepare data =======
        test_data = day_data.copy()
        test_data.reset_index(inplace=True, drop=True)

        # ======= II. Make predictions =======
        X_test = test_data[self.cols_order].copy()

        predictions = self.predictor.predict(X_test=X_test)
        test_data['signal'] = predictions

        return test_data
    