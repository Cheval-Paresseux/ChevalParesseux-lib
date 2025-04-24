from .info import valor_ponto
from .common import strategy
from ...Data_Processing import Prepping as prep
from ...utils import classificationMetrics as clsmet

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import inspect

def filter_params_for_function(func, param_dict):
    sig = inspect.signature(func)
    valid_keys = sig.parameters.keys()
    return {k: v for k, v in param_dict.items() if k in valid_keys}

def extract_universe(params_grid: dict): 
    # ======= 0. Define recursive function to generate all combinations =======
    def recursive_combine(keys, values, index, current_combination, params_list):
        if index == len(keys):
            # Base case: all parameters have been assigned a value
            params_list.append(current_combination.copy())
            return

        key = keys[index]
        for value in values[index]:
            current_combination[key] = value
            recursive_combine(keys, values, index + 1, current_combination, params_list)

    # ======= I. Initialize variables =======
    keys = list(params_grid.keys())
    values = list(params_grid.values())
    params_list = []

    # ======= II. Generate all combinations =======
    recursive_combine(keys, values, 0, {}, params_list)

    return params_list

#! ==================================================================================== #
#! ================================== ML MODEL ======================================== #
class ML_Model(strategy):
    def __init__(
        self, 
        data: pd.DataFrame = None, 
        n_jobs: int = 1,
        lag: str = "open",
        asset: str = None,
    ):
        """
        Constructor for the ML_Model class.
        
        Parameters:
            - data (pd.DataFrame): Input data for the model, should be raw from the data handler.
            - n_jobs (int): Number of jobs to run in parallel. Default is 1.
            - lag (str): Lag value for the backtest methods. Default is "open".
            - asset (str): Asset name for the backtest methods. Default is None.
        """
        # ======= 0. Input Data =======
        self.data = data
        self.n_jobs = n_jobs
        
        self.lag = lag
        self.asset = asset
        
        # ======= I. Data Processing models =======
        self.sampling_model = None
        self.sampling_params = None
        
        self.labeller_model = None
        self.labeller_params = None
        
        self.features_model = None
        self.features_params = None
        
        self.cleaner_model = None
        self.cleaner = None # Store the instance of the cleaner to use for the test data
        self.cleaner_params = None
        
        # ======= II. Tuning Models =======
        self.featuresSelector_model = None
        self.featuresSelector = None # Store the instance of the features selector to use for the test data
        self.featuresSelector_params = None
        
        self.splitSample_model = None
        self.splitSample_params = None
        
        self.gridSearch_model = None
        self.gridSearch_params = None
        self.gridUniverse = None
        
        self.filterGridUniverse = None
        
        # ======= III. Core Models =======
        self.predictor_model = None
        self.predictor = None # Store the instance of the model to use for the test data
        self.predictor_params = None
        
        self.filter_model = None
        self.filter = None # Store the instance of the filter to use for the test data
        self.filter_params = None
        
        # ======= IV. Annexe =======
        self.non_feature_columns = None
        self.price_name = None
        
        # ======= V. Features Information (not inputed) =======
        self.features_informations = None
        self.features_rules = None
        self.metrics = [] # Store the metrics for each backtest
        
        # ======= VI. Pre-computed data =======
        self.preC_test = None # Pre-computed data for the test data
        
    #?__________________________ Initialization Methods __________________________________ #
    def set_models(
        self,
        # Data Processing models
        sampling_model = None,
        labeller_model = None,
        features_model = None,
        cleaner_model = None,
        # Tuning Models
        featuresSelector_model = None,
        splitSample_model = None,
        gridSearch_model = None,
        # Core Models
        predictor_model = None,
        filter_model = None,
    ):
        """
        Set the models to be used in the strategy.
        
        Parameters:
        """
        # ======= I. Data Processing models =======
        self.sampling_model = sampling_model if sampling_model is not None else self.sampling_model
        self.labeller_model = labeller_model if labeller_model is not None else self.labeller_model
        self.features_model = features_model if features_model is not None else self.features_model
        self.cleaner_model = cleaner_model if cleaner_model is not None else self.cleaner_model
        
        # ======= II. Tuning Models =======
        self.featuresSelector_model = featuresSelector_model if featuresSelector_model is not None else self.featuresSelector_model
        self.splitSample_model = splitSample_model if splitSample_model is not None else self.splitSample_model
        self.gridSearch_model = gridSearch_model if gridSearch_model is not None else self.gridSearch_model
        
        # ======= III. Core Models =======
        self.predictor_model = predictor_model if predictor_model is not None else self.predictor_model
        self.filter_model = filter_model if filter_model is not None else self.filter_model
        
        return self
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        # Data Processing models
        sampling_params: dict = None,
        labeller_params: dict = None,
        features_params: dict = None,
        cleaner_params: dict = None,
        # Tuning Models
        featuresSelector_params: dict = None,
        splitSample_params: dict = None,
        gridSearch_params: dict = None,
        gridUniverse: dict = None,
        filterGridUniverse: dict = None,
        # Annexe
        non_feature_columns: list = None,
        price_name: str = None,
    ):
        """
        Set the parameters to be used for all the models.
        
        Parameters:
        """
        # ======= I. Data Processing models =======
        self.sampling_params = sampling_params if sampling_params is not None else self.sampling_params
        self.labeller_params = labeller_params if labeller_params is not None else self.labeller_params
        self.features_params = features_params if features_params is not None else self.features_params
        self.cleaner_params = cleaner_params if cleaner_params is not None else self.cleaner_params
        
        # ======= II. Tuning Models =======
        self.featuresSelector_params = featuresSelector_params if featuresSelector_params is not None else self.featuresSelector_params
        self.splitSample_params = splitSample_params if splitSample_params is not None else self.splitSample_params
        self.gridSearch_params = gridSearch_params if gridSearch_params is not None else self.gridSearch_params
        self.gridUniverse = gridUniverse if gridUniverse is not None else self.gridUniverse
        self.filterGridUniverse = filterGridUniverse if filterGridUniverse is not None else self.filterGridUniverse
        
        # ====== III. Annexe =======
        self.non_feature_columns = non_feature_columns if non_feature_columns is not None else self.non_feature_columns
        self.price_name = price_name if price_name is not None else self.price_name
        
        return self
    
    #?___________________________ Training Processing ____________________________________ #
    def bars_sampling(
        self,
        df: pd.DataFrame,
    ):
        """
        Performs data cumsum resampling based on the specified parameters.
        
        Parameters:
            - df (pd.DataFrame): Input DataFrame to be resampled.
        """
        # ======= I. Initialize Model =======
        sampler = self.sampling_model(data=df, n_jobs=self.n_jobs)
        sampler.set_params(**filter_params_for_function(sampler.set_params, self.sampling_params))
        
        # ======= II. Resample Data =======
        resampled_data = sampler.extract()
        
        return resampled_data
    
    #?____________________________________________________________________________________ #
    def apply_labelling(
        self, 
        df: pd.DataFrame,
        series_name: str = "close",
    ):
        """
        This function applies the labelling model to the data.
        
        Parameters:
            - df (pd.DataFrame): Input DataFrame to be labelled.
            - series_name (str): Name of the column to be used for labelling. Default is "close".
        
        Notes : The parameters should be passed as a dictionary of lists, then only the first set of parameters will be used.
                This comes from the fact that the labelling model allows to compute multiple labels at once.
        """
        # ======= I. Copy the DataFrame =======
        aux_df = df.copy()
        series = aux_df[series_name]
        
        # ======= II. Extract Labels =======
        labeller = self.labeller_model(series=series, n_jobs=self.n_jobs)
        labeller.set_params(**filter_params_for_function(labeller.set_params, self.labeller_params))
        labels_df = labeller.extract()
        labels_series = labels_df["set_0"] # Keep only the first set of labels
        
        # ======= III. Concatenate the Labels with the Original Data =======
        aux_df.loc[:, "label"] = labels_series
        
        return aux_df
    
    #?____________________________________________________________________________________ #
    def apply_features(
        self,
        df: pd.DataFrame,
        series_name: str = "close",
    ):
        """
        Performs the feature extraction on the data.
        
        Parmameters:
            - df (pd.DataFrame): Input DataFrame to be processed.
            - series_name (str): Name of the column to be used for feature extraction. Default is "close".
            
        Notes : The parameters should be passed as a dictionary of lists, then all parameters sets are computed and used as a feature.
        """
        # ======= I. Copy the DataFrame =======
        aux_df = df.copy()
        series = aux_df[series_name]

        # ======= II. Extract Features =======
        features_df = pd.DataFrame()
        for feature_model in self.features_model:
            feature = feature_model(data=series, n_jobs=self.n_jobs)
            feature.set_params(**filter_params_for_function(feature.set_params, self.features_params))

            features_df = pd.concat([features_df, feature.extract()], axis=1)

        # ======= III. Concatenate the Features with the Original Data =======
        aux_df = pd.concat([aux_df, features_df], axis=1)

        return aux_df
    
    #?____________________________________________________________________________________ #
    def clean_data(
        self,
        dfs_list: list,
    ):
        """
        Cleans the data using the cleaner model. It manage nans and outliers in the middle of the data by a forward fill to avoid leakage and performs a normalization of the features.
        
        Parameters:
            - dfs_list (list): List of DataFrames to be cleaned.
        
        Returns: 
            - stacked_data (pd.DataFrame): Stacked DataFrame with all the data cleaned.
            - processed_data (list): List of all the clean dataframes.
            - features_informations (pd.DataFrame): DataFrame with the features information.
            - features_rules (dictionary): Dictionary containing the rules for each feature; (mean and std for the normalization, threshold to consider an outlier => used on test data).
        """
        # ======= I. Initialize and set Parameters =======
        cleaner = self.cleaner_model(training_data=dfs_list, non_feature_columns=self.non_feature_columns, n_jobs=self.n_jobs)
        cleaner.set_params(**filter_params_for_function(cleaner.set_params, self.cleaner_params))
        
        # ======= II. Clean the data =======
        stacked_data, processed_data, features_informations, features_rules = cleaner.extract()
        
        # ======= III. Store the features information =======
        self.features_informations = features_informations
        self.features_rules = features_rules
        self.cleaner = cleaner
        
        return stacked_data, processed_data, features_informations
    
    #?____________________________________________________________________________________ #
    def get_training_data(
        self,
        training_data: pd.DataFrame,
    ):
        """
        Generates the training data for the model by applying the sampling, labelling, feature extraction and cleaning methods.
        
        Parameters:
            - training_data (pd.DataFrame): Input raw DataFrame to be used for training.
        
        Returns:
            - stacked_data (pd.DataFrame): Stacked DataFrame with all the data cleaned.
            - processed_data (list): List of all the clean dataframes.
            - features_informations (pd.DataFrame): DataFrame with the features information.
        """
        # ======= I. Resample Data =======
        resampled_dfs = self.bars_sampling(df=training_data)

        # ======= II. Apply Labelling =======
        labels_dfs = []
        print('Labelling data...')
        for day_df in tqdm(resampled_dfs):
            try:
                label_df = self.apply_labelling(df=day_df, series_name=self.price_name)
                labels_dfs.append(label_df)
            except Exception as e:
                print(f"Skipping labelling due to error: {e}")
                continue
            
        # ======= III. Apply Features =======
        features_dfs = []
        print('Extracting features...')
        for day_df in tqdm(labels_dfs):
            try:
                feature_df = self.apply_features(df=day_df, series_name=self.price_name)
                # Filter the dfs that are too small
                if len(feature_df) > 20:
                    features_dfs.append(feature_df)
            except Exception as e:
                print(f"Skipping feature extraction due to error: {e}")
                continue

        # ======= IV. Clean Data =======
        stacked_data, processed_data, features_informations = self.clean_data(dfs_list=features_dfs)
        
        return stacked_data, processed_data, features_informations
    
    #?____________________________________________________________________________________ #
    def split_and_resample(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        random_state: int = 72,
    ):
        """
        Generate samples for the training data using the sample generator model.
        
        Parameters:
            - df (pd.DataFrame): Input DataFrame to be used for generating samples.
            - n_folds (int): Number of folds to be used for the sample generation. Default is 5.
        
        Returns:
            - folds (list): List of DataFrames with the generated samples.
            - balanced_folds (list): List of DataFrames with the generated samples that have been resampled to account for labels imbalance and importance.
        """
        # ======= I. Initialize =======
        splitSample_model = self.splitSample_model(training_df=df, n_jobs=self.n_jobs, random_state=random_state)
        splitSample_model.set_params(**filter_params_for_function(splitSample_model.set_params, self.splitSample_params))
        
        # ====== II. Generate Samples =======
        folds, balanced_folds = splitSample_model.extract(df=df, n_folds=n_folds)

        return folds, balanced_folds
    
    #?______________________________ Model Fitting _______________________________________ #
    def features_selection(
        self, 
        training_df: pd.DataFrame
    ):
        # ======= I. Extract Features =======
        features_df = training_df.drop(columns=self.non_feature_columns).copy()
        non_features_df = training_df[self.non_feature_columns].copy()
        
        # ======= II. Initialize Selector Model =======
        features_selector = self.featuresSelector_model()
        features_selector.set_params(**filter_params_for_function(features_selector.set_params, self.featuresSelector_params))
        
        # ======= III. Extract New Features df =======
        new_features_df = features_selector.extract(features_df)
        new_df = pd.concat([non_features_df, new_features_df], axis=1)
        
        # ======= IV. Store the Features Selector =======
        self.featuresSelector = features_selector
        
        return new_df
    
    #?____________________________________________________________________________________ #
    def tune_predictor(
        self, 
        training_df: pd.DataFrame,
    ):
        """
        Performs a grid search and cross validation to find the best parameters for the model and fit the model.
        
        Parameters:
        """
        # ======= I. Initialize Grid =======
        grid = self.gridSearch_model(model=self.predictor_model, training_data=training_df, n_jobs=self.n_jobs)
        grid.set_params(
            splitSample_model=self.splitSample_model, 
            splitSample_params=self.splitSample_params, 
            gridSearch_params=self.gridSearch_params, 
            gridUniverse=self.gridUniverse,
            non_feature_columns=self.non_feature_columns
        )

        # ======= II. Find the Best Parameters =======
        best_params, best_score = grid.fit()
        print(f'Best score : {best_score} for params : {best_params}')
        
        # ======= III. Store Parameters =======
        self.predictor_params = best_params
        
        return best_params
    
    #?____________________________________________________________________________________ #
    def fit(
        self, 
        training_df: pd.DataFrame,
    ):
        # ======= I. Generate the folds =======
        balanced_training = self.gridSearch_params["balanced_training"]
        n_folds = self.gridSearch_params["n_folds"]
        
        _, balanced_folds = self.split_and_resample(df=training_df, n_folds=n_folds)
        
        # ======= II. Extract the Training Sample =======
        if balanced_training:
            full_training_df = pd.concat(balanced_folds, ignore_index=True, axis=0).reset_index(drop=True)
        else:
            full_training_df = training_df.copy()
            
        X_train = full_training_df.drop(columns=self.non_feature_columns).copy()
        y_train = full_training_df["label"].copy()
        
        # ======= IV. Fit the Model =======
        predictor = self.predictor_model(n_jobs=self.n_jobs)
        predictor.set_params(**self.predictor_params)
        predictor.fit(X_train, y_train)
        
        self.predictor = predictor

        return self
    
    #?____________________________________________________________________________________ #
    def tune_filter(
        self, 
        processed_data: list,
        costs: float = 0.0,
    ):
        params_list = extract_universe(self.filterGridUniverse)
        
        best_profit = -np.inf
        best_params = None
        profit_history = []
        for params in tqdm(params_list):
            # ======= I. Initialize the Filter =======
            filter = self.filter_model()
            filter.set_params(**filter_params_for_function(filter.set_params, params))
            self.filter = filter

            # ======= II. Run Backtest with the current Filter =======
            results = []
            self.metrics = [] # Reset the metrics for each backtest, they are computed each time we call get_signals
            for df in processed_data:
                df_ops = self.predict(df)
                if not df_ops.empty:
                    results.append(df_ops)
                
            operations = pd.concat(results)
            operations['Profit'] -= 2 * costs * operations['Size']
            
            # ======= III. Keep the Best Params =======
            profit_total = operations['Profit'].sum()
            profit_history.append({"params": params, "profit": profit_total}) 
            
            if profit_total > best_profit:
                best_profit = profit_total
                best_params = params
                best_metrics = self.metrics.copy()
        
        # ======= IV. Store the Best Parameters =======
        filter = self.filter_model()
        filter.set_params(**filter_params_for_function(filter.set_params, best_params))
        
        self.filter_params = best_params
        self.filter = filter
        self.metrics = best_metrics
        
        return profit_history

    #?___________________________ Backtest Methods _______________________________________ #
    def transform_data(
        self, 
        df: pd.DataFrame
    ):
        """
        Applies a series of transformations to the data, including resampling, labelling, feature extraction, and cleaning.
        This method is called when backtesting the strategy so it is used only on test data.
        
        Parameters:
            - df (pd.DataFrame): Input DataFrame to be transformed.
        
        Returns:
            - clean_dfs (list): List of DataFrames with the transformed data.
        """
        # ======= 0. If the data is already pre-computed, return directly =======
        if self.preC_test is not None:
            test_data = self.preC_test.copy()
            
            print('Selecting features...')
            new_data = []
            for df in tqdm(test_data):
                features_df = df.drop(columns=self.non_feature_columns).copy()
                non_features_df = df[self.non_feature_columns].copy()
                
                new_features_df = self.featuresSelector.extract_new(features_df)
                new_df = pd.concat([non_features_df, new_features_df], axis=1)
                new_data.append(new_df)
                
            return new_data
        
        # ======= I. Resample Data =======
        resampled_dfs = self.bars_sampling(df=df)

        # ======= II. Apply Labelling =======
        labels_dfs = []
        print('Labelling data...')
        for day_df in tqdm(resampled_dfs):
            try:
                label_df = self.apply_labelling(df=day_df, series_name=self.price_name)
                labels_dfs.append(label_df)
            except Exception as e:
                print(f"Skipping labelling due to error: {e}")
                continue
            
        # ======= III. Apply Features =======
        features_dfs = []
        print('Extracting features...')
        for day_df in tqdm(labels_dfs):
            try:
                feature_df = self.apply_features(df=day_df, series_name=self.price_name)
                # Filter the dfs that are too small
                if len(feature_df) > 20:
                    features_dfs.append(feature_df)
            except Exception as e:
                print(f"Skipping feature extraction due to error: {e}")
                continue
        
        # ======= IV. Clean Data =======
        clean_dfs = []
        print('Cleaning data...')
        for df in tqdm(features_dfs):
            clean_df = self.cleaner.extract_new(new_data=df)
            if len(clean_df) > 5:
                clean_dfs.append(clean_df)
                
        print(f'{len(clean_dfs)} dataframes cleaned')
        
        # ======= V. features Selection =======
        print('Selecting features...')
        new_data = []
        for df in tqdm(clean_dfs):
            features_df = df.drop(columns=self.non_feature_columns).copy()
            non_features_df = df[self.non_feature_columns].copy()
            
            new_features_df = self.featuresSelector.extract_new(features_df)
            new_df = pd.concat([non_features_df, new_features_df], axis=1)
            new_data.append(new_df)
        
        return new_data

    #?____________________________________________________________________________________ #
    def get_signals(
        self, 
        df: pd.DataFrame
    ):
        """
        Uses the model to predict the signals on the test data.
        
        Parameters:
            - df (pd.DataFrame): Input DataFrame to be used for predictions.
        
        Returns:
            - results_df (pd.DataFrame): DataFrame with the predicted signals.
        """
        # ======= 0. Initialization =======
        results_df = df.copy()
        
        X_test = results_df.drop(columns=self.non_feature_columns).copy()
        y_test = results_df["label"].copy()
        
        # ======= II. Make Predictions =======
        predictions = self.predictor.predict(X_test)
        predictions = pd.Series(predictions, index=X_test.index)
        
        # ======= III. Filter the Signals =======
        predictions_filtered = self.filter.extract(predictions)
        
        results_df.loc[X_test.index, "signal"] = predictions_filtered
        
        # ======= IV. Compute Stats =======
        for_stats_df = results_df[["signal", "label"]].copy().fillna(0)
        metrics = clsmet.get_classification_metrics(for_stats_df["signal"], for_stats_df["label"])
        self.metrics.append(metrics)

        return results_df
    
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        df: pd.DataFrame
    ):
        """
        Transforms a dataframe containing signals into a dataframe of operations.
        
        Parameters:
            - df (list): List of DataFrames to be used for predictions.
        
        Returns:
            - operations (pd.DataFrame): DataFrame with the operations generated from the signals.
        """
        # ======= I. Initialize Operations =======
        operations = pd.DataFrame(columns=["ID", "Trade Type", "Entry ts", "Entry price", "Exit ts", "Exit price", "Size", "Profit"])
        
        # ======== II. Extract Signals =======
        try:
            signals_df = self.get_signals(df=df)
        except Exception as e:
            print(f"Skipping predict due to error: {e}")
            return operations
        
        # ======= II. Generate Operations from the signals =======
        # II.1 Adjust Last Prices (issue in the data input)
        signals_df.loc[signals_df.index[-1], "bid_open"] = signals_df.loc[signals_df.index[-1], "close"]
        signals_df.loc[signals_df.index[-1], "ask_open"] = signals_df.loc[signals_df.index[-1], "close"]

        # II.2 Set first and last signal to 0 to ensure that the operations are closed
        signals_df.reset_index(drop=True, inplace=True)
        signals_df.loc[0, "signal"] = 0
        signals_df.loc[len(signals_df) - 1, "signal"] = 0
        signals_df.loc[len(signals_df) - 2, "signal"] = 0
        
        signals_df = signals_df.ffill()

        # II.3 Extract the Signal Change and the Entry Points
        signals_df["signal_change"] = signals_df["signal"].diff()
        signals_df["signal_change"] = signals_df["signal_change"].shift(1)

        entry_points = signals_df[signals_df["signal_change"] != 0]

        # ======= III. Create an Operation for each entry point =======
        entry_points = entry_points.copy()
        entry_points_size = len(entry_points)
        sequential_id = 0

        for idx in range(entry_points_size - 1):
            # III.1 Extracting rows
            current_row = entry_points.iloc[idx]
            next_row = entry_points.iloc[idx + 1]
            row_before_current_row = signals_df.iloc[current_row.name - 1]

            # III.2 Extract Information for a Long Operation
            if (current_row["signal_change"] > 0 and row_before_current_row["signal"] == 1):
                trade_type = "buy"
                entry_ts = current_row["ts_open"]
                entry_price = current_row["ask_open"]
                exit_ts = next_row["ts_open"]
                exit_price = next_row["bid_open"]

                size = 1  # signals_df.loc[1, "size"]
                profit = (exit_price - entry_price) * valor_ponto[self.asset]

            # III.3 Extract Information for a Short Operation
            elif (current_row["signal_change"] < 0 and row_before_current_row["signal"] == -1):
                trade_type = "sell"
                entry_ts = current_row["ts_open"]
                entry_price = current_row["bid_open"]
                exit_ts = next_row["ts_open"]
                exit_price = next_row["ask_open"]

                size = 1  # signals_df.loc[1, "size"]

                profit = (entry_price - exit_price) * valor_ponto[self.asset]

            else:
                continue

            # III.4 Append Operation to the DataFrame
            operations.loc[sequential_id] = [sequential_id, trade_type, entry_ts, entry_price, exit_ts, exit_price, size, profit]
            sequential_id += 1

        return operations
    

#! ==================================================================================== #
#! ================================== LABELLER ======================================== #
class Labeller_Benchmark(strategy):
    def __init__(
        self, 
        data: pd.DataFrame = None, 
        n_jobs: int = 1,
        lag: str = "open",
        asset: str = None,
    ):
        """
        Constructor for the ML_Model class.
        
        Parameters:
            - data (pd.DataFrame): Input data for the model, should be raw from the data handler.
            - n_jobs (int): Number of jobs to run in parallel. Default is 1.
            - lag (str): Lag value for the backtest methods. Default is "open".
            - asset (str): Asset name for the backtest methods. Default is None.
        """
        # ======= I. Input Data =======
        self.data = data
        self.n_jobs = n_jobs
        
        # ======= II. Data Processing models =======
        # ----> Core models
        self.predictive_model = None
        self.model = None # Store the instance of the model to use for the test data
        
        self.filter_model = None
        
        # ----> Training models
        self.labeller_model = None
        self.features_model = None
        
        # ----> Data Cleaning models
        self.cleaner_model = None
        self.cleaner = None # Store the instance of the cleaner to use for the test data
        
        # ----> Grid Search models
        self.sampleGenerator_model = None
        self.gridSearch_model = None
        
        # ======= III. Models Parameters =======
        # ----> Core models
        self.predictive_params = None
        self.filter_params = None
        
        # ----> Training models
        self.labeller_params = None
        self.features_params = None
        
        # ----> Data Cleaning models
        self.cleaner_params = None
        self.pre_threshold = None
        
        # ----> Grid Search models
        self.sampling_params = None
        self.sampleGenerator_params = None
        self.gridSearch_params = None
        self.params_filter_grid = None
        
        # ----> Columns to not be used as features
        self.non_feature_columns = None
        
        # ======= IV. Features Information =======
        self.features_informations = None
        self.features_rules = None
        self.metrics = [] # Store the metrics for each backtest

        # ======= IV. Backtest Parameters =======
        self.lag = lag
        self.asset = asset

    #?__________________________ Initialization Methods __________________________________ #
    def set_models(
        self,
        labeller_model,
        cleaner_model,
    ):
        """
        Set the models to be used in the strategy.
        
        Parameters:
            - labeller_model: Labeller model to be used for labelling data.                    ===> GrandLine/Data_Processing/Labelling
            - cleaner_model: Data cleaning model to be used.                                   ===> GrandLine/Data_Processing/Prepping
        """
        self.labeller_model = labeller_model
        self.cleaner_model = cleaner_model
    
        return self
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        labeller_params: dict,
        sampling_params: dict,
        cleaner_params: dict,
        non_feature_columns: list = ["open","high","low","close","volume","ts_open","ts_close","date","bid_open","ask_open"],
    ):
        """
        Set the parameters to be used for all the models.
        
        Parameters:
            - labeller_params (dict): Parameters for the labeller model.
                Ex : (slope_labeller) | labeller_params = {'horizon': [5], 'horizon_extension': [1.5], 'min_trend_size': [10], 'smoothing_method': ["ewma"], 'window_smooth': [10], 'lambda_smooth': [0.2]}
                
            - sampling_params (dict): Parameters for the sampling model.
                Ex : (extract_targetBars) | sampling_params = {"target_bars": 100, "column_name": "volume", "window_bars_estimation": 10, "new_cols_methods": "mean", "grouping_column": "date"}
            
            - cleaner_params (dict): Parameters for the data cleaning model.
                Ex : (DataCleaner) | cleaner_params = {'stationarity_threshold': 0.05, 'outliers_threshold': 5, 'mean_tolerance': 0.05, 'std_tolerance': 0.05}

            - non_feature_columns (list): List of columns to be excluded from the features.
                Ex : non_feature_columns = ["ts_open", "ts_close", "open", "high", "low", "close", "volume", "date", "bid_open", "ask_open", "label"]
        """
        self.labeller_params = labeller_params
        self.sampling_params = sampling_params
        self.cleaner_params = cleaner_params
        self.non_feature_columns = non_feature_columns
        
        self.pre_threshold = sampling_params["target_bars"]
        
        return self
    
    #?_______________________________ Helper Methods _____________________________________ #
    def bars_sampling(
        self,
        df: pd.DataFrame,
    ):
        """
        Performs data cumsum resampling based on the specified parameters.
        
        Parameters:
            - df (pd.DataFrame): Input DataFrame to be resampled.
        """
        # ===== I. Resample Data =======
        resampled_dfs, last_threshold = prep.extract_targetBars(
            data=df,
            target_bars=self.sampling_params["target_bars"],
            column_name=self.sampling_params["column_name"],
            window_bars_estimation=self.sampling_params["window_bars_estimation"],
            new_cols_methods=self.sampling_params["new_cols_methods"],
            grouping_column=self.sampling_params["grouping_column"],
            pre_threshold=self.pre_threshold,
            n_jobs=self.n_jobs,
        )
        
        self.pre_threshold = last_threshold
        
        # ===== II. Get Rid of the first days (incorrect estimation) =======
        window_bars_estimation = self.sampling_params["window_bars_estimation"]
        resampled_dfs = resampled_dfs[window_bars_estimation:]
        
        return resampled_dfs
    
    #?____________________________________________________________________________________ #
    def apply_labelling(
        self, 
        df: pd.DataFrame,
        series_name: str = "close",
    ):
        """
        This function applies the labelling model to the data.
        
        Parameters:
            - df (pd.DataFrame): Input DataFrame to be labelled.
            - series_name (str): Name of the column to be used for labelling. Default is "close".
        
        Notes : The parameters should be passed as a dictionary of lists, then only the first set of parameters will be used.
                This comes from the fact that the labelling model allows to compute multiple labels at once.
        """
        # ======= I. Copy the DataFrame =======
        aux_df = df.copy()
        series = aux_df[series_name]
        
        # ======= II. Extract Labels =======
        labeller = self.labeller_model(series=series, n_jobs=self.n_jobs).set_params(**self.labeller_params)
        labels_df = labeller.extract()
        labels_series = labels_df["set_0"] # Keep only the first set of labels
        
        # ======= III. Concatenate the Labels with the Original Data =======
        aux_df.loc[:, "label"] = labels_series
        
        return aux_df
    
    #?____________________________________________________________________________________ #
    def clean_data(
        self,
        dfs_list: list,
    ):
        """
        Cleans the data using the cleaner model. It manage nans and outliers in the middle of the data by a forward fill to avoid leakage and performs a normalization of the features.
        
        Parameters:
            - dfs_list (list): List of DataFrames to be cleaned.
        
        Returns: 
            - stacked_data (pd.DataFrame): Stacked DataFrame with all the data cleaned.
            - processed_data (list): List of all the clean dataframes.
            - features_informations (pd.DataFrame): DataFrame with the features information.
            - features_rules (dictionary): Dictionary containing the rules for each feature; (mean and std for the normalization, threshold to consider an outlier => used on test data).
        """
        # ======= I. Initialize and set Parameters =======
        cleaner = self.cleaner_model(training_data=dfs_list, non_feature_columns=self.non_feature_columns, n_jobs=self.n_jobs)
        cleaner.set_params(**self.cleaner_params)
        
        # ======= II. Clean the data =======
        stacked_data, processed_data, features_informations, features_rules = cleaner.extract()
        stacked_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        stacked_data = stacked_data.ffill(inplace=False).copy()
        
        # ======= III. Store the features information =======
        self.features_informations = features_informations
        self.features_rules = features_rules
        self.cleaner = cleaner
        
        return stacked_data, processed_data, features_informations
    
    #?__________________________ Preparation Methods _____________________________________ #
    def get_training_data(
        self,
        training_data: pd.DataFrame,
    ):
        """
        Generates the training data for the model by applying the sampling, labelling, feature extraction and cleaning methods.
        
        Parameters:
            - training_data (pd.DataFrame): Input raw DataFrame to be used for training.
        
        Returns:
            - stacked_data (pd.DataFrame): Stacked DataFrame with all the data cleaned.
            - processed_data (list): List of all the clean dataframes.
            - features_informations (pd.DataFrame): DataFrame with the features information.
        """
        # ======= I. Resample Data =======
        resampled_dfs = self.bars_sampling(df=training_data)

        # ======= II. Apply Labelling =======
        labels_dfs = []
        print('Labelling data...')
        for day_df in tqdm(resampled_dfs):
            try:
                label_df = self.apply_labelling(df=day_df, series_name=self.sampleGenerator_params['price_name'])
                labels_dfs.append(label_df)
            except Exception as e:
                print(f"Skipping labelling due to error: {e}")
                continue

        # ======= IV. Clean Data =======
        stacked_data, processed_data, features_informations = self.clean_data(dfs_list=labels_dfs)
        
        return stacked_data, processed_data, features_informations
    
    #?____________________________________________________________________________________ #
    def fit(
        self, 
    ):
        pass

    #?___________________________ Backtest Methods _______________________________________ #
    def transform_data(
        self, 
        df: pd.DataFrame
    ):
        """
        Applies a series of transformations to the data, including resampling, labelling, feature extraction, and cleaning.
        This method is called when backtesting the strategy so it is used only on test data.
        
        Parameters:
            - df (pd.DataFrame): Input DataFrame to be transformed.
        
        Returns:
            - clean_dfs (list): List of DataFrames with the transformed data.
        """
        # ======= I. Resample Data =======
        resampled_dfs = self.bars_sampling(df=df)

        # ======= II. Apply Labelling =======
        labels_dfs = []
        print('Labelling data...')
        for day_df in tqdm(resampled_dfs):
            label_df = self.apply_labelling(df=day_df, series_name=self.sampleGenerator_params['price_name'])
            labels_dfs.append(label_df)
        
        # ======= IV. Clean Data =======
        clean_dfs = []
        print('Cleaning data...')
        for df in tqdm(labels_dfs):
            clean_df = self.cleaner.extract_new(new_data=df)
            if len(clean_df) > 10:
                clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                clean_df = clean_df.ffill(inplace=False).copy()
                clean_dfs.append(clean_df)
                
        print(f'{len(clean_dfs)} dataframes cleaned')
        
        return clean_dfs

    #?____________________________________________________________________________________ #
    def get_signals(
        self, 
        df: pd.DataFrame
    ):
        """
        Uses the model to predict the signals on the test data.
        
        Parameters:
            - df (pd.DataFrame): Input DataFrame to be used for predictions.
        
        Returns:
            - results_df (pd.DataFrame): DataFrame with the predicted signals.
        """
        # ======= 0. Initialization =======
        results_df = df.copy()
        results_df['signal'] = results_df['label'].copy()

        return results_df
    
    #?____________________________________________________________________________________ #
    def filter_signals(
        self, 
        signals: pd.Series
    ):
        return signals

    #?____________________________________________________________________________________ #
    def predict(
        self, 
        df: pd.DataFrame
    ):
        """
        Transforms a dataframe containing signals into a dataframe of operations.
        
        Parameters:
            - df (list): List of DataFrames to be used for predictions.
        
        Returns:
            - operations (pd.DataFrame): DataFrame with the operations generated from the signals.
        """
        # ======= I. Initialize Operations =======
        operations = pd.DataFrame(columns=["ID", "Trade Type", "Entry ts", "Entry price", "Exit ts", "Exit price", "Size", "Profit"])
        
        # ======== II. Extract Signals =======
        try:
            signals_df = self.get_signals(df=df)
        except Exception as e:
            print(f"Skipping predict due to error: {e}")
            return operations
        
        # ======= II. Generate Operations from the signals =======
        # II.1 Adjust Last Prices (issue in the data input)
        signals_df.loc[signals_df.index[-1], "bid_open"] = signals_df.loc[signals_df.index[-1], "close"]
        signals_df.loc[signals_df.index[-1], "ask_open"] = signals_df.loc[signals_df.index[-1], "close"]

        # II.2 Set first and last signal to 0 to ensure that the operations are closed
        signals_df.reset_index(drop=True, inplace=True)
        signals_df.loc[0, "signal"] = 0
        signals_df.loc[len(signals_df) - 1, "signal"] = 0
        signals_df.loc[len(signals_df) - 2, "signal"] = 0
        
        signals_df = signals_df.ffill()

        # II.3 Extract the Signal Change and the Entry Points
        signals_df["signal_change"] = signals_df["signal"].diff()
        signals_df["signal_change"] = signals_df["signal_change"].shift(1)

        entry_points = signals_df[signals_df["signal_change"] != 0]

        # ======= III. Create an Operation for each entry point =======
        entry_points = entry_points.copy()
        entry_points_size = len(entry_points)
        sequential_id = 0

        for idx in range(entry_points_size - 1):
            # III.1 Extracting rows
            current_row = entry_points.iloc[idx]
            next_row = entry_points.iloc[idx + 1]
            row_before_current_row = signals_df.iloc[current_row.name - 1]

            # III.2 Extract Information for a Long Operation
            if (current_row["signal_change"] > 0 and row_before_current_row["signal"] == 1):
                trade_type = "buy"
                entry_ts = current_row["ts_open"]
                entry_price = current_row["ask_open"]
                exit_ts = next_row["ts_open"]
                exit_price = next_row["bid_open"]

                size = 1  # signals_df.loc[1, "size"]
                profit = (exit_price - entry_price) * valor_ponto[self.asset]

            # III.3 Extract Information for a Short Operation
            elif (current_row["signal_change"] < 0 and row_before_current_row["signal"] == -1):
                trade_type = "sell"
                entry_ts = current_row["ts_open"]
                entry_price = current_row["bid_open"]
                exit_ts = next_row["ts_open"]
                exit_price = next_row["ask_open"]

                size = 1  # signals_df.loc[1, "size"]

                profit = (entry_price - exit_price) * valor_ponto[self.asset]

            else:
                continue

            # III.4 Append Operation to the DataFrame
            operations.loc[sequential_id] = [sequential_id, trade_type, entry_ts, entry_price, exit_ts, exit_price, size, profit]
            sequential_id += 1

        return operations
    
