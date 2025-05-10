import numpy as np
import pandas as pd

#! ==================================================================================== #
#! =============================== TRINARY LABELLERS ================================== #
class tripleBarrier_labeller(com.Labeller):
    """
    Triple Barrier Method for labelling time series data.
    
    This class computes labels based on the first barrier hit (upper, lower, or vertical) within a specified time window. 
    It inherits from the Labeller base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_labels : compute the moving average feature over a rolling window
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ):
        """
        Initializes the TripleBarrierLabeller with a number of jobs.
        
        Parameters:
            - n_jobs (int): The number of jobs to run in parallel. Default is 1.
        """
        super().__init__(
            n_jobs=n_jobs,
        )
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        upper_barrier: list = [0.5, 1, 2, 3],
        lower_barrier: list = [0.5, 1, 2, 3],
        vertical_barrier: list = [5, 10, 15, 20, 25, 30],
        vol_window: list = [5, 10, 15, 30],
        smoothing_method: list = [None, "ewma", "average"],
        window_smooth: list = [5, 10, 15],
        lambda_smooth: list = [0.2, 0.5, 0.7],
    ):
        """
        Defines the parameters grid for the TripleBarrierLabeller.
        
        Parameters:
            - upper_barrier (list): The upper barrier for the label.
            - lower_barrier (list): The lower barrier for the label.
            - vertical_barrier (list): The vertical barrier for the label.
            - vol_window (list): The window size for the volatility calculation.
            - smoothing_method (list): The smoothing method to be applied. Options are "ewma" or "average".
            - window_smooth (list): The window size for the smoothing method. It should a number of bars.
            - lambda_smooth (list): The lambda parameter for the ewma method. It should be in [0, 1].
        """
        self.params = {
            "upper_barrier": upper_barrier,
            "lower_barrier": lower_barrier,
            "vertical_barrier": vertical_barrier,
            "vol_window": vol_window,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }

        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: pd.Series,
    ):
        """
        Applies preprocessing to the input data before labels extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        ________
        N.B: The labeller does not require preprocessing, but this method is kept for consistency.
        """
        return data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        data: pd.Series,
        upper_barrier: float,
        lower_barrier: float,
        vertical_barrier: int,
        vol_window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes the triple barrier labels of the processed series.

        Parameters: 
            - data (pd.Series): The input series to be processed.
            - upper_barrier (float): The upper barrier for the label.
            - lower_barrier (float): The lower barrier for the label.
            - vertical_barrier (int): The vertical barrier for the label.
            - vol_window (int): The window size for the volatility calculation.
            - smoothing_method (str): The smoothing method to be applied. Options are "ewma" or "average".
            - window_smooth (int): The window size for the smoothing method. It should a number of bars.
            - lambda_smooth (float): The lambda parameter for the ewma method. It should be in [0, 1].
            
        Returns:
            - labels_series (pd.Series): A series of {-1, 0, 1} labels based on future performance.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Compute the Volatility Series =======
        returns_series = processed_series.pct_change().fillna(0)
        volatility_series = returns_series.rolling(vol_window).std() * np.sqrt(vol_window)

        # ======= III. Initialize the labeled series and trade side =======
        labels_series = pd.Series(index=processed_series.index, dtype=int)
        trade_side = 0

        # ======= IV. Iterate through the price series =======
        for index in processed_series.index:
            # IV.1 Extract the future prices over the horizon
            start_idx = processed_series.index.get_loc(index)
            end_idx = min(start_idx + vertical_barrier, len(processed_series))
            future_prices = processed_series.iloc[start_idx:end_idx]

            # IV.2 Compute the range of future returns over the horizon
            max_price = future_prices.max()
            min_price = future_prices.min()

            max_price_index = future_prices.idxmax()
            min_price_index = future_prices.idxmin()

            max_return = (max_price - processed_series.loc[index]) / processed_series.loc[index]
            min_return = (min_price - processed_series.loc[index]) / processed_series.loc[index]

            # IV.3 Adjust the barrier thresholds with the volatility
            upper_threshold = upper_barrier * volatility_series.loc[index]
            lower_threshold = lower_barrier * volatility_series.loc[index]

            # IV.4 Check if the horizontal barriers have been hit
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

            # IV.5 Label based on the first event that occurs
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

            # IV.6 Update the trade side
            trade_side = labels_series.loc[index]

        return labels_series

#*____________________________________________________________________________________ #
class lookForward_labeller(com.Labeller):
    """
    Look-Forward Labelling Method for time series data.
    
    This class labels data based on the future return over a look-ahead window, relative to its expected volatility. 
    It inherits from the Labeller base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_labels : compute the moving average feature over a rolling window
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ):
        """
        Initializes the LookForwardLabeller with a time series and number of jobs.
        
        Parameters:
            - n_jobs (int): The number of jobs to run in parallel. Default is 1.
        """
        super().__init__(
            n_jobs=n_jobs,
        )
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        window_lookForward: list = [5, 10, 15],
        min_trend_size: list = [5, 10, 15],
        volatility_threshold: list = [0.5, 1, 1.5, 2, 2.5, 3],
        smoothing_method: list = [None, "ewma", "average"],
        window_smooth: list = [5, 10, 15],
        lambda_smooth: list = [0.2, 0.5, 0.7],
    ):
        """
        Defines the parameter grid for the LookForwardLabeller.
        
        Parameters:
            - window_lookForward (list): Look-ahead window in bars to compute future returns.
            - min_trend_size (list): Minimum duration (in bars) that a trend must persist to be labeled.
            - volatility_threshold (list): Threshold used to compare return-to-volatility ratio.
            - smoothing_method (list): Optional smoothing technique ("ewma", "average", or None).
            - window_smooth (list): Size of the smoothing window, if applied.
            - lambda_smooth (list): Decay factor for EWMA smoothing. Values should be in [0, 1].
        """
        self.params = {
            "window_lookForward": window_lookForward,
            "min_trend_size": min_trend_size,
            "volatility_threshold": volatility_threshold,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }

        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: pd.Series,
    ):
        """
        Applies preprocessing to the input data before labels extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        ________
        N.B: The labeller does not require preprocessing, but this method is kept for consistency.
        """
        return data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        data: pd.Series,
        window_lookForward: int,
        min_trend_size: int,
        volatility_threshold: float,
        smoothing_method: str,
        lambda_smooth: float,
        window_smooth: int,
    ):
        """
        Computes labels based on look-ahead returns and volatility ratios.

        Parameters:
            - data (pd.Series): The input series to be processed.
            - window_lookForward (int): Number of bars to look ahead for return calculation.
            - min_trend_size (int): Minimum number of consecutive identical labels required.
            - volatility_threshold (float): Minimum return-to-volatility ratio for a label.
            - smoothing_method (str): Type of smoothing applied to the input series.
            - lambda_smooth (float): Decay factor for EWMA smoothing.
            - window_smooth (int): Size of smoothing window.

        Returns:
            - labels_series (pd.Series): A series of {-1, 0, 1} labels based on future performance.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= I. Significant look forward Label =======
        # ------- 1. Get the moving X days returns and the moving X days volatility -------
        Xdays_returns = (processed_series.shift(-window_lookForward) - processed_series) / processed_series
        Xdays_vol = Xdays_returns.rolling(window=window_lookForward).std()

        # ------- 2. Compare the X days returns to the volatility  -------
        Xdays_score = Xdays_returns / Xdays_vol
        Xdays_label = Xdays_score.apply(lambda x: 1 if x > volatility_threshold else (-1 if x < -volatility_threshold else 0))

        # ------- 3. Eliminate the trends that are too small -------
        labels_series = fil.segment_length_filter(label_series=Xdays_label, window=min_trend_size)

        return labels_series

#*____________________________________________________________________________________ #
class regR2rank_labeller(com.Labeller):
    """
    Regression R² Rank Labeller for time series data.
    
    This class labels data points based on the strength of linear trends  detected via rolling regression windows. 
    It compares the R² of the linear fit over a range of horizons to a threshold, and assigns labels based on trend direction and significance.
    It inherits from the Labeller base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_labels : compute the moving average feature over a rolling window
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ):
        """
        Initializes the regR2rank_labeller with a time series and number of jobs.
        
        Parameters:
            - n_jobs (int): The number of jobs to run in parallel. Default is 1.
        """
        super().__init__(
            n_jobs=n_jobs,
        )
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        horizon: list = [5, 10, 15],
        horizon_extension: list = [1.1, 1.5, 2],
        r2_threshold: list = [0.1, 0.5, 0.7],
        min_trend_size: list = [5, 10, 15],
        smoothing_method: list = [None, "ewma", "average"],
        window_smooth: list = [5, 10, 15],
        lambda_smooth: list = [0.2, 0.5, 0.7],
    ):
        """
        Defines the parameter grid for the regR2rank_labeller.
        
        Parameters:
            - horizon (list): Minimum number of future bars to evaluate the trend.
            - horizon_extension (list): Multiplicative factor to extend the horizon range.
            - r2_threshold (list): R² threshold to qualify a trend as significant.
            - min_trend_size (list): Minimum consecutive bars for a trend to be retained.
            - smoothing_method (list): Smoothing method to preprocess the series. Options are "ewma" or "average".
            - window_smooth (list): Window size used for smoothing.
            - lambda_smooth (list): Lambda parameter for the EWMA smoothing method.
        """
        self.params = {
            "horizon": horizon,
            "horizon_extension": horizon_extension,
            "r2_threshold": r2_threshold,
            "min_trend_size": min_trend_size,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }

        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: pd.Series,
    ):
        """
        Applies preprocessing to the input data before labels extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        ________
        N.B: The labeller does not require preprocessing, but this method is kept for consistency.
        """
        return data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        data: pd.Series,
        horizon: int,
        horizon_extension: float,
        r2_threshold: float,
        min_trend_size: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Computes labels by detecting statistically significant linear trends via R² score.
        
        Parameters:
            - data (pd.Series): The input series to be processed.
            - horizon (int): Minimum look-ahead period (number of bars).
            - horizon_extension (float): Factor to extend the range of look-ahead periods.
            - r2_threshold (float): Threshold for R² value to consider a trend significant.
            - min_trend_size (int): Minimum trend length required to retain label.
            - smoothing_method (str): Smoothing technique to apply before computing labels.
            - window_smooth (int): Size of the smoothing window.
            - lambda_smooth (float): Lambda value for EWMA smoothing.
        
        Returns:
            - labels_series (pd.Series): A pd.Series of -1, 0, or 1 labels based on trend direction.
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()

        # ======= II. Labelling Process =======
        nb_elements = len(processed_series)
        labels_series = pd.Series(0, index=processed_series.index, dtype=int)

        horizon_max = round(horizon * (1 + horizon_extension))
        for idx in range(nb_elements - horizon + 1):
            # III.0 Skip the NaN values
            if pd.isna(processed_series.iloc[idx]): 
                continue

            # III.1 Iterate over different horizons to find the most significant trend
            best_r2 = 0
            for current_horizon in range(horizon, horizon_max):
                # ------ 1. Extract the future EMA values ------
                future_ewma = processed_series.iloc[idx:idx + current_horizon]
                temporality = np.arange(len(future_ewma))  

                # ------ 2. Fit the Linear Regression and Extract R² ------
                model = reg.OLSRegression()
                model.fit(temporality, future_ewma)
                r2 = model.statistics["R_squared"]
                slope = model.coefficients[0]

                # ------ 3. Check if the trend is significant ------
                if r2 > best_r2 and r2 > r2_threshold:
                    best_r2 = r2
                    labels_series.iloc[idx] = 1 if slope > 0 else -1  
        
        # ======= III. Eliminate the trends that are too small =======
        labels_series = fil.segment_length_filter(label_series=labels_series, window=min_trend_size)

        return labels_series

#*____________________________________________________________________________________ #
class boostedlF_labeller(com.Labeller):
    """
    Boosted Look-Forward Labeller for time series data.

    This ensemble labeller combines signals from both a look-forward volatility-based labeller and a regression-based trend strength labeller. 
    It inherits from the Labeller base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_labels : compute the moving average feature over a rolling window
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ):
        """
        Initialize the boostedlF_labeller.

        Parameters:
            - n_jobs (int): Number of jobs for parallel processing (default is 1).
        """
        super().__init__(
            n_jobs=n_jobs,
        )
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        window_lookForward: list = [5, 10, 15],
        min_trend_size: list = [5, 10, 15],
        volatility_threshold: list = [0.5, 1, 1.5, 2, 2.5, 3],
        horizon: list = [5, 10, 15],
        horizon_extension: list = [1.1, 1.5, 2],
        r2_threshold: list = [0.1, 0.5, 0.7],
        trend_size: list = [5, 10, 15],
        smoothing_method: list = [None, "ewma", "average"],
        window_smooth: list = [5, 10, 15],
        lambda_smooth: list = [0.2, 0.5, 0.7],
    ):
        """
        Sets the parameter grid for the boosted look-forward labeller.

        Parameters:
            - window_lookForward (list): Look-forward window size to identify trends.
            - min_trend_size (list): Minimum size of a continuous trend.
            - volatility_threshold (list): Minimum volatility needed to validate a trend.
            - horizon (list): Regression horizon (forward window size).
            - horizon_extension (list): Multiplier to extend horizon during regression.
            - r2_threshold (list): Minimum R² to qualify regression as significant.
            - trend_size (list): Final trend filter window to remove short-lived signals.
            - smoothing_method (list): Type of smoothing to apply to data ("ewma" or "average").
            - window_smooth (list): Smoothing window length.
            - lambda_smooth (list): EWMA decay factor (between 0 and 1).
        """
        self.params = {
            "window_lookForward": window_lookForward,
            "min_trend_size": min_trend_size,
            "volatility_threshold": volatility_threshold,
            "horizon": horizon,
            "horizon_extension": horizon_extension,
            "r2_threshold": r2_threshold,
            "trend_size": trend_size,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }

        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: pd.Series,
    ):
        """
        Applies preprocessing to the input data before labels extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        ________
        N.B: The labeller does not require preprocessing, but this method is kept for consistency.
        """
        return data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        data: pd.Series,
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
        """
        Combines look-forward and regression labellers to generate a boosted label series.

        Parameters:
            - data (pd.Series): The input series to be processed.
            - window_lookForward (int): Look-ahead window for the look-forward labeller.
            - min_trend_size (int): Minimum trend length for both labellers.
            - volatility_threshold (float): Volatility threshold for the look-forward labeller.
            - horizon (int): Minimum horizon for regression.
            - horizon_extension (float): Multiplier to extend regression horizon.
            - r2_threshold (float): Threshold for R² to confirm a trend.
            - trend_size (int): Final trend size filter window.
            - smoothing_method (str): Smoothing technique applied to both labellers.
            - window_smooth (int): Window size used for smoothing.
            - lambda_smooth (float): EWMA smoothing lambda.

        Returns:
            - labels_series (pd.Series): Final filtered label series combining both methods.
        """
        # ======= I. Preprocess =======        
        processed_series = self.process_data(data=data).dropna()

        # ======= I. Extract Labels =======
        lF_labeller = lookForward_labeller(n_jobs=self.n_jobs)
        lF_labeller.set_params(
            window_lookForward=[window_lookForward], 
            min_trend_size=[min_trend_size], 
            volatility_threshold=[volatility_threshold], 
            smoothing_method=[smoothing_method], 
            window_smooth=[window_smooth], 
            lambda_smooth=[lambda_smooth]
        )
        r2_labeller = regR2rank_labeller(n_jobs=self.n_jobs)
        r2_labeller.set_params(
            horizon=[horizon],
            horizon_extension=[horizon_extension],
            r2_threshold=[r2_threshold],
            min_trend_size=[min_trend_size],
            smoothing_method=[smoothing_method],
            window_smooth=[window_smooth],
            lambda_smooth=[lambda_smooth],
        )

        lF_labels = lF_labeller.extract(data=processed_series)
        r2_labels = r2_labeller.extract(data=processed_series)
        
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
        ensemble_labels = ensemble_labels.ffill()
        ensemble_labels = ensemble_labels.replace(2, 1).replace(-2, -1).replace(3, 1).replace(-3, -1)

        # ------- 2. Manage the case of direct change in trend in reg_label -------
        mask_positive_to_negative = (ensemble_labels == 1) & (r2_labels == -1)
        mask_negative_to_positive = (ensemble_labels == -1) & (r2_labels == 1)
        ensemble_labels[mask_positive_to_negative | mask_negative_to_positive] = 0

        # ------- 3. Eliminate the trends that are too small -------
        labels_series = fil.segment_length_filter(label_series=ensemble_labels, window=trend_size)

        # ------- 4. Eliminate the last point of each trend -------
        next_label = labels_series.shift(-1)
        labels_series[next_label == 0] = 0

        return labels_series



#! ==================================================================================== #
#! =============================== BINARY LABELLERS =================================== #
class slope_labeller(com.Labeller):
    """
    Slope-Based Labeller for time series data.

    This binary labeller detects up/down trends by fitting linear regression on future price segments and selecting the steepest slope.
    It inherits from the Labeller base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_labels : compute the moving average feature over a rolling window
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ):
        """
        Initialize the slope_labeller.

        Parameters:
            - n_jobs (int): Number of jobs for parallel processing (default is 1).
        """
        super().__init__(
            n_jobs=n_jobs,
        )
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        horizon: list = [5, 10, 15],
        horizon_extension: list = [1.1, 1.5, 2],
        min_trend_size: list = [5, 10, 15],
        smoothing_method: list = [None, "ewma", "average"],
        window_smooth: list = [5, 10, 15],
        lambda_smooth: list = [0.2, 0.5, 0.7],
    ):
        """
        Sets the parameter grid for the slope-based labeller.

        Parameters:
            - horizon (list): Number of forward bars used for trend detection.
            - horizon_extension (list): Multiplier to extend max future horizon.
            - min_trend_size (list): Minimum continuous trend length allowed.
            - smoothing_method (list): Optional smoothing ("ewma", "average", or None).
            - window_smooth (list): Number of bars for smoothing window.
            - lambda_smooth (list): Decay factor for EWMA smoothing (0 < λ ≤ 1).
        """
        self.params = {
            "horizon": horizon,
            "horizon_extension": horizon_extension,
            "min_trend_size": min_trend_size,
            "smoothing_method": smoothing_method,
            "window_smooth": window_smooth,
            "lambda_smooth": lambda_smooth,
        }

        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: pd.Series,
    ):
        """
        Applies preprocessing to the input data before labels extraction.
        
        Parameters:
            - data (pd.Series): The input data to be processed.
        
        Returns:
            - processed_data (pd.Series): The smoothed series, or raw series if no smoothing is applied.
        ________
        N.B: The labeller does not require preprocessing, but this method is kept for consistency.
        """
        return data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        data: pd.Series,
        horizon: int,
        horizon_extension: float,
        min_trend_size: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        """
        Assigns +1/-1 binary labels based on best slope over a forward window.

        Parameters:
            - data (pd.Series): The input series to be processed.
            - horizon (int): Base number of bars to look forward for trend estimation.
            - horizon_extension (float): Multiplier to stretch the max horizon.
            - min_trend_size (int): Minimum trend length to retain a label.
            - smoothing_method (str): Optional smoothing type before labelling.
            - window_smooth (int): Smoothing window size.
            - lambda_smooth (float): Lambda for EWMA smoothing.

        Returns:
            - labels_series (pd.Series): Binary series with +1 (up), -1 (down) or NaN (no label).
        """
        # ======= I. Smooth the Data & Preprocess =======
        smoothed_series = self.smooth_data(
            data=data, 
            smoothing_method=smoothing_method, 
            window_smooth=window_smooth, 
            lambda_smooth=lambda_smooth
        )
        
        processed_series = self.process_data(data=smoothed_series).dropna()
        
        # ======= II. Labelling Process =======
        nb_elements = len(processed_series)
        labels_series = pd.Series(0, index=processed_series.index, dtype=int)

        horizon_max = round(horizon * (1 + horizon_extension))
        for idx in range(nb_elements - horizon + 1):
            # II.0 Skip the NaN values
            if pd.isna(processed_series.iloc[idx]):
                continue

            # II.1 Iterate over different horizons to find the most significant trend
            best_slope = 0
            for current_horizon in range(horizon, horizon_max):
                # ------ 1. Extract the future EMA values ------
                future_ewma = processed_series.iloc[idx : idx + current_horizon]
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
        labels_series = fil.segment_length_filter(label_series=labels_series, window=min_trend_size)
        labels_series = labels_series.replace(0, np.nan).ffill()

        return labels_series

