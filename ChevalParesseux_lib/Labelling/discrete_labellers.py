from ..Measures import Filters as fil
from ..Models import linearRegression as reg
from ..Labelling import common as com

import numpy as np
import pandas as pd

#! ==================================================================================== #
#! =============================== TRINARY LABELLERS ================================== #
class tripleBarrier_labeller(com.Labeller):
    def __init__(
        self, 
        data: pd.Series, 
        params: dict = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "upper_barrier": [0.2, 0.5, 1, 2, 3, 5, 10],
                "lower_barrier": [0.2, 0.5, 1, 2, 3, 5, 10],
                "vertical_barrier": [5, 10, 15, 20, 25, 30],
                "window": [5, 10, 15, 20, 25, 30],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            params=params,
            n_jobs=n_jobs,
            )
    
    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        upper_barrier: float,
        lower_barrier: float,
        vertical_barrier: int,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute volatility target =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        returns_series = series.pct_change().fillna(0)
        volatility_series = returns_series.rolling(window).std() * np.sqrt(window)

        # ======= II. Initialize the labeled series and trade side =======
        labels_series = pd.Series(index=series.index, dtype=int)
        trade_side = 0

        # ======= III. Iterate through the price series =======
        for index in series.index:
            # III.1 Extract the future prices over the horizon
            start_idx = series.index.get_loc(index)
            end_idx = min(start_idx + vertical_barrier, len(series))
            future_prices = series.iloc[start_idx:end_idx]

            # III.2 Compute the range of future returns over the horizon
            max_price = future_prices.max()
            min_price = future_prices.min()

            max_price_index = future_prices.idxmax()
            min_price_index = future_prices.idxmin()

            max_return = (max_price - series.loc[index]) / series.loc[index]
            min_return = (min_price - series.loc[index]) / series.loc[index]

            # III.3 Adjust the barrier thresholds with the volatility
            upper_threshold = upper_barrier * volatility_series.loc[index]
            lower_threshold = lower_barrier * volatility_series.loc[index]

            # III.4 Check if the horizontal barriers have been hit
            long_event = False
            short_event = False

            if trade_side == 1:  # Long trade
                if max_return > upper_threshold:
                    long_event = True
                elif min_return < -lower_threshold:
                    short_event = True

            elif trade_side == -1:  # Short trade
                if min_return < -upper_threshold:
                    short_event = True
                elif max_return > lower_threshold:
                    long_event = True

            else:  # No position held
                if max_return > upper_threshold:
                    long_event = True
                elif min_return < -upper_threshold:
                    short_event = True

            # III.5 Label based on the first event that occurs
            if long_event and short_event:  # If both events occur, choose the first one
                if max_price_index < min_price_index:
                    labels_series.loc[index] = 1
                else:
                    labels_series.loc[index] = -1

            elif long_event and not short_event:  # If only long event occurs
                labels_series.loc[index] = 1

            elif short_event and not long_event:  # If only short event occurs
                labels_series.loc[index] = -1

            else:  # If no event occurs (vertical hit)
                labels_series.loc[index] = 0

            # III.6 Update the trade side
            trade_side = labels_series.loc[index]

        return labels_series

#*____________________________________________________________________________________ #
class lookForward_labeller(com.Labeller):
    def __init__(
        self, 
        data: pd.Series, 
        params: dict, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window_lookForward": [5, 10, 15, 20, 25, 30],
                "min_trend_size": [5, 10, 15, 20, 25, 30],
                "volatility_threshold": [0.5, 1, 1.5, 2, 2.5, 3],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            params=params,
            n_jobs=n_jobs,
            )
    
    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        window_lookForward: int,
        min_trend_size: int,
        volatility_threshold: float,
        smoothing_method: str,
        lambda_smooth: float,
        window_smooth: int,
    ):
        # ======= I. Prepare Series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()

        # ======= I. Significant look forward Label =======
        # ------- 1. Get the moving X days returns and the moving X days volatility -------
        Xdays_returns = (series.shift(-window_lookForward) - series) / series
        Xdays_vol = Xdays_returns.rolling(window=window_lookForward).std()

        # ------- 2. Compare the X days returns to the volatility  -------
        Xdays_score = Xdays_returns / Xdays_vol
        Xdays_label = Xdays_score.apply(lambda x: 1 if x > volatility_threshold else (-1 if x < -volatility_threshold else 0))

        # ------- 3. Eliminate the trends that are too small -------
        labels_series = com.trend_filter(label_series=Xdays_label, window=min_trend_size)

        return labels_series

#*____________________________________________________________________________________ #
class regR2rank_labeller(com.Labeller):
    def __init__(
        self, 
        data: pd.Series, 
        params: dict, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "horizon": [5, 10, 15, 20, 25, 30],
                "horizon_extension": [1.1, 1.3, 1.5, 2, 2.5, 3],
                "r2_threshold": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                "trend_size": [5, 10, 15, 20, 25, 30],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            params=params,
            n_jobs=n_jobs,
            )
    
    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        horizon: int,
        horizon_extension: float,
        r2_threshold: float,
        trend_size: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Prepare Series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        nb_elements = len(series)

        labels_series = pd.Series(0, index=series.index, dtype=int)  # Initialise à 0

        # ======= II. Labelling Process =======
        horizon_max = round(horizon * (1 + horizon_extension))
        for idx in range(nb_elements - horizon + 1):
            # III.0 Skip the NaN values
            if pd.isna(series.iloc[idx]):  # Correction ici
                continue

            # III.1 Iterate over different horizons to find the most significant trend
            best_r2 = 0
            for current_horizon in range(horizon, horizon_max):
                # ------ 1. Extract the future EMA values ------
                future_ewma = series.iloc[idx:idx + current_horizon]
                temporality = np.arange(len(future_ewma))  # Correction ici

                # ------ 2. Fit the Linear Regression and Extract R² ------
                model = reg.OLSRegression()
                model.fit(temporality, future_ewma)
                r2 = model.statistics["R_squared"]
                slope = model.coefficients[0]

                # ------ 3. Check if the trend is significant ------
                if r2 > best_r2 and r2 > r2_threshold:
                    best_r2 = r2
                    labels_series.iloc[idx] = 1 if slope > 0 else -1  # Correction ici
        
        # ======= III. Eliminate the trends that are too small =======
        labels_series = com.trend_filter(label_series=labels_series, window=trend_size)

        return labels_series

#*____________________________________________________________________________________ #
class boostedlF_labeller(com.Labeller):
    def __init__(
        self, 
        data: pd.Series, 
        params: dict, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                # ------- LookForward -------
                "window_lookForward": [5, 10, 15, 20, 25, 30],
                "min_trend_size": [5, 10, 15, 20, 25, 30],
                "volatility_threshold": [0.5, 1, 1.5, 2, 2.5, 3],
                # ------- regR2rank -------
                "horizon": [5, 10, 15, 20, 25, 30],
                "horizon_extension": [1.1, 1.3, 1.5, 2, 2.5, 3],
                "r2_threshold": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                "trend_size": [5, 10, 15, 20, 25, 30],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            params=params,
            n_jobs=n_jobs,
            )
    
    #?____________________________________________________________________________________ #
    def process_data(self):
        processed_data = self.data
        
        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        window_lookForward: int,
        min_trend_size: int,
        volatility_threshold: float,
        horizon: int,
        horizon_extension: float,
        r2_threshold: float,
        trend_size: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Extract Labels =======
        lF_labeller = lookForward_labeller(
            data=self.data, 
            params={
                "window_lookForward": [window_lookForward], 
                "min_trend_size": [min_trend_size], 
                "volatility_threshold": [volatility_threshold], 
                "smoothing_method": [smoothing_method], 
                "window_smooth": [window_smooth], 
                "lambda_smooth": [lambda_smooth]
            }
        )
        r2_labeller = regR2rank_labeller(
            data=self.data, 
            params={
                "horizon": [horizon], 
                "horizon_extension": [horizon_extension], 
                "r2_threshold": [r2_threshold], 
                "trend_size": [trend_size], 
                "smoothing_method": [smoothing_method], 
                "window_smooth": [window_smooth], 
                "lambda_smooth": [lambda_smooth]
            }
        )
        lF_labels = lF_labeller.extract()
        r2_labels = r2_labeller.extract()
        
        # ======= II. Linking Trend Holes in regR2rank =======
        r2_labels = r2_labels.replace(0, np.nan)
        forward = r2_labels.ffill()
        backward = r2_labels.bfill()
        r2_labels = forward + backward
        r2_labels = r2_labels.replace(1, 0).replace(-1, 0).replace(2, 1).replace(-2, -1)

        # ======= III. Labels Ensemble =======
        # ------- 1. Combine the labels using lookForward as base -------
        ensemble_labels = lF_labels * 2 + r2_labels
        ensemble_labels = ensemble_labels.replace(1, np.nan).replace(-1, np.nan)
        ensemble_labels = ensemble_labels.fillna(method="ffill")
        ensemble_labels = ensemble_labels.replace(2, 1).replace(-2, -1).replace(3, 1).replace(-3, -1)

        # ------- 2. Manage the case of direct change in trend in reg_label -------
        mask_positive_to_negative = (ensemble_labels == 1) & (r2_labels == -1)
        mask_negative_to_positive = (ensemble_labels == -1) & (r2_labels == 1)
        ensemble_labels[mask_positive_to_negative | mask_negative_to_positive] = 0

        # ------- 3. Eliminate the trends that are too small -------
        labels_series = com.trend_filter(label_series=ensemble_labels, window=trend_size)

        # ------- 4. Eliminate the last point of each trend -------
        next_label = labels_series.shift(-1)
        labels_series[next_label == 0] = 0

        return labels_series



#! ==================================================================================== #
#! =============================== BINARY LABELLERS =================================== #
class slope_labeller(com.Labeller):
    def __init__(
        self, 
        data: pd.Series, 
        params: dict = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "horizon": [5, 10, 15, 20, 25, 30],
                "horizon_extension": [1.1, 1.3, 1.5, 2, 2.5, 3],
                "trend_size": [5, 10, 15, 20, 25, 30],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            params=params,
            n_jobs=n_jobs,
            )
    
    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        horizon: int,
        horizon_extension: float,
        trend_size: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Prepare Series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        nb_elements = len(series)

        labels_series = pd.Series(0, index=series.index, dtype=int)

        # ======= II. Labelling Process =======
        horizon_max = round(horizon * (1 + horizon_extension))
        for idx in range(nb_elements - horizon + 1):
            # II.0 Skip the NaN values
            if pd.isna(series.iloc[idx]):
                continue

            # II.1 Iterate over different horizons to find the most significant trend
            best_slope = 0
            for current_horizon in range(horizon, horizon_max):
                # ------ 1. Extract the future EMA values ------
                future_ewma = series.iloc[idx : idx + current_horizon]
                temporality = np.arange(len(future_ewma))

                # ------ 2. Fit the Linear Regression and Extract R² ------
                model = reg.OLSRegression()
                model.fit(temporality, future_ewma)
                slope = model.coefficients[0]

                # ------ 3. Check if the trend is significant ------
                if abs(slope) > abs(best_slope):
                    best_slope = slope
                    labels_series.iloc[idx] = 1 if slope > 0 else -1

        # ======= III. Eliminate the trends that are too small =======
        labels_series = com.trend_filter(label_series=labels_series, window=trend_size)
        labels_series = labels_series.replace(0, np.nan).ffill()

        return labels_series

#*____________________________________________________________________________________ #

