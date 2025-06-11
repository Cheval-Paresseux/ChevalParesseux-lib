import pandas as pd
import numpy as np
from typing import Union, List
from joblib import Parallel, delayed



class OpenToClose_Strategy():
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(
        self
    ) -> None:
        # ======= I. Strategy Parameters =======
        self.params = {}
        self.selector = None
        self.predictor = None
        self.cols_order = None

        # ======= II. Backtest Parameters =======
        self.lag = None
        self.asset = None
        self.n_jobs = 1

        return None

    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        #_____ features Parameters _____ #
        features_models: dict,
        #_____ labeller Parameters _____ #
        labeller_model: tuple,
        #_____ Fitting Parameters _____ #
        resampler_params: dict,
        selector_params: dict,
        tuner_params: dict,
        grid_universe: list,
        grid_criteria: str,
        #_____ backtest Parameters _____ #
        lag: str,
        asset: str,
        valor_ponto: float,
        n_jobs: int = 10
    ):
        # ======= I. Strategy Parameters =======
        self.params = {
            #_____ features Parameters _____ #
            'features_models': features_models,
            #_____ labeller Parameters _____ #
            'labeller_model': labeller_model,
            #_____ Fitting Parameters _____ #
            'resampler_params': resampler_params,
            'selector_params': selector_params,
            'tuner_params': tuner_params,
            'grid_universe': grid_universe,
            'grid_criteria': grid_criteria,
        }
        
        # ======= II. Backtest Parameters =======
        self.lag = lag
        self.asset = asset
        self.valor_ponto = valor_ponto
        self.n_jobs = n_jobs

        return self
    
    #?_______________________________ Processing methods _________________________________ #
    def get_series_decomposition(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Performs series decomposition on the provided time series data to extract intraday, overnight, and complete returns.
        
        Parameters:
            - data (pd.DataFrame): DataFrame containing the time series data with columns 'date', 'open', 'close', 'high', 'low'.
        
        Returns:
            - intraday_df (pd.DataFrame): DataFrame containing intraday returns and related metrics.
            - overnight_df (pd.DataFrame): DataFrame containing overnight returns and related metrics.
            - complete_df (pd.DataFrame): DataFrame containing complete returns and related metrics.
        """
        # ======= I. Prepare Data =======
        daily_data = [group for _, group in data.groupby('date')]
        
        dates = sorted(data['date'].unique())
        intraday_df = pd.DataFrame(index=dates, columns=['returns', 'returns_high', 'returns_low', 'open', 'close', 'high', 'low'])
        overnight_df = pd.DataFrame(index=dates, columns=['returns', 'returns_high', 'returns_low', 'open', 'close', 'high', 'low'])
        complete_df = pd.DataFrame(index=dates, columns=['returns', 'returns_high', 'returns_low', 'open', 'close', 'high', 'low'])
        
        # --- 1. Intraday Data ---
        intraday_df = intraday_df.astype(float)
        intraday_df.iloc[0, :3] = 0 # returns for the first day are set to 0, we initialize with the original values
        intraday_df.loc[intraday_df.index[0], 'open'] = daily_data[0]['open'].iloc[0] # The open of the day
        intraday_df.loc[intraday_df.index[0], 'close'] = daily_data[0]['close'].iloc[-1] # The close of the day
        intraday_df.loc[intraday_df.index[0], 'high'] = daily_data[0]['high'].max() # The high of the day
        intraday_df.loc[intraday_df.index[0], 'low'] = daily_data[0]['low'].min() # The low of the day

        # --- 2. Overnight Data ---
        overnight_df = overnight_df.astype(float)
        overnight_df.iloc[0, :3] = 0 # returns for the first day are set to 0, we initialize with the original values
        overnight_df.loc[intraday_df.index[0], 'open'] = daily_data[0]['close'].iloc[-1] # The open of the overnight is the close of the intraday
        overnight_df.loc[intraday_df.index[0], 'close'] = daily_data[1]['open'].iloc[0] # The close of the overnight is the open of the next day
        overnight_df.loc[intraday_df.index[0], 'high'] = max(daily_data[0]['close'].iloc[-1], daily_data[1]['open'].iloc[0])
        overnight_df.loc[intraday_df.index[0], 'low'] = min(daily_data[0]['close'].iloc[-1], daily_data[1]['open'].iloc[0])

        # --- 3. Complete Data ---
        complete_df = complete_df.astype(float)
        complete_df.iloc[0, :3] = 0 # returns for the first day are set to 0, we initialize with the original values
        complete_df.loc[intraday_df.index[0], 'open'] = daily_data[0]['open'].iloc[0] # The open of the day
        complete_df.loc[intraday_df.index[0], 'close'] = daily_data[1]['open'].iloc[0] # The close of the day is the open of the next day
        complete_df.loc[intraday_df.index[0], 'high'] = max(daily_data[0]['high'].max(), daily_data[1]['open'].iloc[0]) # The high of the day is the max of the highs of the day and th open of the next day
        complete_df.loc[intraday_df.index[0], 'low'] = min(daily_data[0]['low'].min(), daily_data[1]['open'].iloc[0]) # The low of the day is the min of the lows of the day and th open of the next day
        
        # ======= II. Extract returns =======
        for i in range(1, len(daily_data) - 1):
            # --- 1. Get daily data ---
            day_df = daily_data[i].copy()
            date = day_df['date'].iloc[0]
            next_day_open = daily_data[i + 1]['open'].iloc[0] if i < len(daily_data) - 1 else np.nan
            
            # --- 2. Calculate Intra-day Returns ---
            open_intra = day_df['open'].iloc[0]
            close_intra = day_df['close'].iloc[-1]
            high_intra = day_df['high'].max()
            low_intra = day_df['low'].min()
            
            returns_intra = (close_intra - open_intra) / open_intra if open_intra != 0 else np.nan
            returns_intra_high = (high_intra - open_intra) / open_intra if open_intra != 0 else np.nan
            returns_intra_low = (low_intra - open_intra) / open_intra if open_intra != 0 else np.nan
            
            intraday_df.loc[date, ['returns', 'returns_high', 'returns_low']] = [returns_intra, returns_intra_high, returns_intra_low]
            intraday_df.loc[date, 'open'] = intraday_df['close'].iloc[i -1]
            intraday_df.loc[date, 'close'] = intraday_df.loc[date, 'open'] * (1 + returns_intra)
            intraday_df.loc[date, 'high'] = intraday_df.loc[date, 'open'] * (1 + returns_intra_high)
            intraday_df.loc[date, 'low'] = intraday_df.loc[date, 'open'] * (1 + returns_intra_low)
            
            intraday_df.dropna(inplace=True)
            intraday_df.drop(columns=['returns_high', 'returns_low'], inplace=True)
            
            # --- 3. Calculate Overnight Returns ---
            open_overnight = day_df['close'].iloc[-1] if i < len(daily_data) - 1 else np.nan
            close_overnight = next_day_open
            high_overnight = np.max([day_df['close'].iloc[-1], next_day_open]) if i < len(daily_data) - 1 else np.nan
            low_overnight = np.min([day_df['close'].iloc[-1], next_day_open]) if i < len(daily_data) - 1 else np.nan
            
            returns_overnight = (close_overnight - open_overnight) / open_overnight if open_overnight != 0 else np.nan
            returns_overnight_high = (high_overnight - open_overnight) / open_overnight if open_overnight != 0 else np.nan
            returns_overnight_low = (low_overnight - open_overnight) / open_overnight if open_overnight != 0 else np.nan
            
            overnight_df.loc[date, ['returns', 'returns_high', 'returns_low']] = [returns_overnight, returns_overnight_high, returns_overnight_low]
            overnight_df.loc[date, 'open'] = overnight_df['close'].iloc[i - 1]
            overnight_df.loc[date, 'close'] = overnight_df.loc[date, 'open'] * (1 + returns_overnight)
            overnight_df.loc[date, 'high'] = overnight_df.loc[date, 'open'] * (1 + returns_overnight_high)
            overnight_df.loc[date, 'low'] = overnight_df.loc[date, 'open'] * (1 + returns_overnight_low)
            
            overnight_df.dropna(inplace=True)
            overnight_df.drop(columns=['returns_high', 'returns_low'], inplace=True)
            
            # --- 4. Calculate Complete Returns ---
            open_complete = open_intra
            close_complete = next_day_open
            high_complete = np.max([high_intra, next_day_open]) if i < len(daily_data) - 1 else np.nan
            low_complete = np.min([low_intra, next_day_open]) if i < len(daily_data) - 1 else np.nan
            
            returns_complete = (close_complete - open_complete) / open_complete if open_complete != 0 else np.nan
            returns_complete_high = (high_complete - open_complete) / open_complete if open_complete != 0 else np.nan
            returns_complete_low = (low_complete - open_complete) / open_complete if open_complete != 0 else np.nan
            
            complete_df.loc[date, ['returns', 'returns_high', 'returns_low']] = [returns_complete, returns_complete_high, returns_complete_low]
            complete_df.loc[date, 'open'] = complete_df['close'].iloc[i - 1]
            complete_df.loc[date, 'close'] = complete_df.loc[date, 'open'] * (1 + returns_complete)
            complete_df.loc[date, 'high'] = complete_df.loc[date, 'open'] * (1 + returns_complete_high)
            complete_df.loc[date, 'low'] = complete_df.loc[date, 'open'] * (1 + returns_complete_low)
            
            complete_df.dropna(inplace=True)
            complete_df.drop(columns=['returns_high', 'returns_low'], inplace=True)
        
        return intraday_df, overnight_df, complete_df, daily_data

    #?____________________________________________________________________________________ #
    def process_train_data(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        # ======= I. Decompose Series =======
        intraday_df, overnight_df, complete_df, daily_data = self.get_series_decomposition(data=data)
        features_model = self.params['features_models']
        
        # ======= II. Extract Features from the Series =======
        features_dfs = []
        for series_df in [intraday_df, overnight_df, complete_df]:
            # ----- i. Extract the series from the DataFrame -----
            open_series = series_df['open']
            high_series = series_df['high']
            low_series = series_df['low']
            close_series = series_df['close']
            
            prefix = 'intraday' if series_df is intraday_df else 'overnight' if series_df is overnight_df else 'complete'
            
            # ----- ii. Extract the features for the series -----
            series_features = []
            for feature_name, feature_model in features_model.items():
                model, params = feature_model
                feature = model(n_jobs=self.n_jobs)
                feature.set_params(**params)
                
                if feature_name == "ehlersFisher":
                    feature_df = feature.extract(data=(high_series, low_series))
                elif feature_name == "vortex":
                    feature_df = feature.extract(data=(close_series, high_series, low_series))
                elif feature_name == "vigor":
                    feature_df = feature.extract(data=(open_series, close_series, high_series, low_series))
                elif feature_name == "stochasticOscillator":
                    feature_df = feature.extract(data=(close_series, high_series, low_series))
                else:
                    feature_df = feature.extract(data=close_series)
                
                feature_df.columns = [f'{prefix}_{col}' for col in feature_df.columns]
                series_features.append(feature_df)
            
            # ----- iii. Concatenate features -----
            series_features_df = pd.concat(series_features, axis=1)
            features_dfs.append(series_features_df)

        # ======= III. Concatenate all features into a single DataFrame =======
        features_df = pd.concat(features_dfs, axis=1)
        self.cols_order = features_df.columns
        
        # ======= IV. Extract Labels using Naive Labeller =======
        labeller_model, labeller_params = self.params['labeller_model']
        labeller = labeller_model(n_jobs=self.n_jobs)
        labeller.set_params(**labeller_params)
        
        labels = labeller.extract(data=intraday_df['close'])
        labels_series = labels[labels.columns[0]].copy()
        labels_series = labels_series.rename('label')
        
        # ======= V. Combine Features and Labels into a single DataFrame =======
        intra_close = intraday_df['close']
        intra_close.name = 'intra_close'
        
        processed_data = pd.concat([intra_close, features_df, labels_series], axis=1)
        processed_data['label'] = processed_data['label'].shift(-1) # to align labels with the next row
        processed_data.reset_index(inplace=True)
        processed_data.rename(columns={'index': 'date'}, inplace=True)

        return processed_data
    
    #?____________________________________________________________________________________ #
    def process_data(
        self,
        data: pd.DataFrame,
    ) -> list:
        # ======= I. Decompose Series =======
        intraday_df, overnight_df, complete_df, daily_data = self.get_series_decomposition(data=data)
        features_model = self.params['features_models']
        
        # ======= II. Extract Features from the Series =======
        features_dfs = []
        for series_df in [intraday_df, overnight_df, complete_df]:
            # ----- i. Extract the series from the DataFrame -----
            open_series = series_df['open']
            high_series = series_df['high']
            low_series = series_df['low']
            close_series = series_df['close']
            
            prefix = 'intraday' if series_df is intraday_df else 'overnight' if series_df is overnight_df else 'complete'
            
            # ----- ii. Extract the features for the series -----
            series_features = []
            for feature_name, feature_model in features_model.items():
                model, params = feature_model
                feature = model(n_jobs=self.n_jobs)
                feature.set_params(**params)
                
                if feature_name == "ehlersFisher":
                    feature_df = feature.extract(data=(high_series, low_series))
                elif feature_name == "vortex":
                    feature_df = feature.extract(data=(close_series, high_series, low_series))
                elif feature_name == "vigor":
                    feature_df = feature.extract(data=(open_series, close_series, high_series, low_series))
                elif feature_name == "stochasticOscillator":
                    feature_df = feature.extract(data=(close_series, high_series, low_series))
                else:
                    feature_df = feature.extract(data=close_series)
                
                feature_df.columns = [f'{prefix}_{col}' for col in feature_df.columns]
                series_features.append(feature_df)
            
            # ----- iii. Concatenate features -----
            series_features_df = pd.concat(series_features, axis=1)
            features_dfs.append(series_features_df)

        # ======= III. Concatenate all features into a single DataFrame =======
        features_df = pd.concat(features_dfs, axis=1)
        features_df = features_df[self.cols_order]  # Ensure the same columns order as in training
        
        # ======= IV. Extract Labels using Naive Labeller =======
        labeller_model, labeller_params = self.params['labeller_model']
        labeller = labeller_model(n_jobs=self.n_jobs)
        labeller.set_params(**labeller_params)
        
        labels = labeller.extract(data=intraday_df['close'])
        labels_series = labels[labels.columns[0]].copy()
        labels_series = labels_series.rename('label')
        
        # ======= V. Combine Features and Labels into a single DataFrame =======
        big_df = pd.concat([features_df, labels_series], axis=1)
        big_df['label'] = big_df['label'].shift(-1) # to align labels with the day we actually trade it 
        big_df = big_df.shift(1)  # Shift to align with the actual operation day (as we then merge day by day it is necessary because we won't shift the signals)
        
        # ======= VI. Merge daily data with big_df =======
        big_df = self.selector.extract(data=big_df)
        big_df.reset_index(inplace=True)
        big_df.rename(columns={'index': 'date'}, inplace=True)
        
        # Convert daily_data list into a single DataFrame
        combined_daily_df = pd.concat(daily_data, ignore_index=True)

        # Merge with feature data on date
        enriched_df = combined_daily_df.merge(big_df, on='date', how='left')
        enriched_df = enriched_df.dropna(axis=0)
        daily_data = [group for _, group in enriched_df.groupby('date')]

        return daily_data

    #?________________________________ Fitting methods ___________________________________ #
    def fit(
        self,
        processed_train_data: pd.DataFrame,
    ):
        # ======= 0. Set non-features (always the same if transformed data from transform_data method) =======
        non_features = ['date', 'label', 'intra_close']
        train_data = processed_train_data.dropna(axis=0).copy()
        
        # ======= I. Resample Training Data =======
        resampler = gl.Temporal_uniqueness_selection(n_jobs=self.n_jobs, random_state=72)
        resampler_params = self.params['resampler_params']
        resampler.set_params(**resampler_params)
        
        datasets = resampler.extract(data=train_data)
        resampled_df = datasets[0][0]
        resampled_df.sort_values(by='date', inplace=True)
        resampled_df.reset_index(drop=True, inplace=True)
        
        # ======= II. Feature Selection =======
        features_df = resampled_df.drop(columns=non_features).copy()

        selector = gl.Correlation_selector(n_jobs=self.n_jobs)
        selector_params = self.params['selector_params']
        selector.set_params(**selector_params)

        selector.fit(data=features_df)
        self.selector = selector
        
        train_df = selector.extract(data=resampled_df)

        # ======= III. Fit the predictor =======
        # ----- 1. Tuner Model -----
        tuner = gl.Classifier_gridSearch(n_jobs=self.n_jobs, random_state=72)
        tuner_params = self.params['tuner_params']
        tuner.set_params(**tuner_params)

        # ----- 2. Fitting Model -----
        model = gl.SKL_randomForest_classifier
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
        
        return X_train, y_train
    
    #?________________________________ Predict methods ___________________________________ #
    def get_signals(
        self,
        day_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # ======= I. Prepare data =======
        non_features = ['label', 'ts', 'open', 'high', 'low', 'close', 'volume', 'date', 'bid_open', 'ask_open', 'bid_high', 'bid_low', 'ask_high', 'ask_low', 'number_of_trades', 'number_of_bids', 'number_of_asks']
        test_data = day_data.copy()
        test_data.reset_index(inplace=True, drop=True)

        # ======= II. Make predictions =======
        X_test = test_data.drop(columns=non_features).copy()

        predictions = self.predictor.predict(X_test=X_test)
        test_data['signal'] = predictions

        return test_data
    
