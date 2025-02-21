import numpy as np
import pandas as pd

def pca_weights(cov_matrix: pd.DataFrame, risk_distribution=None, risk_target=1.):
    """
    This function does a PCA decomposition of the covariance matrix and returns the weights of the portfolio that maximizes the Sharpe Ratio.
    
    Args: 
        cov_matrix (pd.DataFrame): Covariance matrix of the assets
        risk_distribution (np.array): Risk distribution of each asset
        risk_target (float): Target risk of the portfolio
    
    Returns:
        weights_by_assets (np.array): Weights of the portfolio
        portfolio_variance (float): Variance of the portfolio
    """
    
    # ======= I. Extract Eigen Values and Vectors =======
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    
    # ======= II. Sort in Descending Order =======
    indices = eigen_values.argsort()[::-1] 
    eigen_values, eigen_vectors = eigen_values[indices], eigen_vectors[:, indices]
    
    # ======= III. Allocate Risk =======
    # III.1 If no risk distribution is given, allocate all to the last PC
    if risk_distribution is None:
        risk_distribution = np.zeros(cov_matrix.shape[0])
        risk_distribution[-1] = 1
        
    # III.2 Normalize risk distribution to sum to Target Risk in PCA space
    weights_by_component = risk_target * np.sqrt((risk_distribution / eigen_values)) # EValues are expressed as variances thus the sqrt
    
    # ======= IV. Projects PC weights into the assets space =======
    weights_by_assets = np.dot(eigen_vectors, np.reshape(weights_by_component, (-1, 1)))
    
    # ======= V. Compute the overall portfolio variance =======
    portfolio_variance = np.dot(weights_by_assets.T, np.dot(cov_matrix, weights_by_assets))[0, 0]
    
    return weights_by_assets, portfolio_variance

def cusum_filter(price_series: pd.Series, threshold: float):
    """
    This function applies the Symmetric CUSUM Filter to a time series and returns the events.
    
    Args:
        time_series (pd.Series): Time series to filter
        threshold (float): Threshold value for the filter
    
    Returns:
        indexed_events (pd.DatetimeIndex): Datetime index of the events
    """
    # ======= I. Initialize Variables =======
    events_list, upward_cumsum, downward_cumsum = [], 0, 0
    diff_series = price_series.diff()
    
    # ======= II. Iterate through the differentiated time series =======
    for index in diff_series.index[1:]:
        # II.1 Update the cumulative sums
        upward_cumsum = max(0, upward_cumsum + diff_series.loc[index])
        downward_cumsum = min(0, downward_cumsum + diff_series.loc[index])
        
        # II.2 Check if the cumulative sums exceed the threshold value
        if downward_cumsum < -threshold:
            downward_cumsum = 0
            events_list.append(index)
            
        elif upward_cumsum > threshold:
            upward_cumsum = 0
            events_list.append(index)
            
    # ======= III. Associate the events with the time series =======
    indexed_events = pd.DatetimeIndex(events_list)
    
    return indexed_events

def rescaled_cusum_filter(price_series: pd.Series, threshold: float):
    """
    This function applies the Symmetric CUSUM Filter to a time series and returns the events.
    
    Args:
        time_series (pd.Series): Time series to filter
        threshold (float): Threshold value for the filter
    
    Returns:
        indexed_events (pd.DatetimeIndex): Datetime index of the events
    """
    # ======= I. Initialize Variables =======
    events_list, upward_cumsum, downward_cumsum = [], 0, 0
    returns_series = price_series.pct_change().fillna(0)
    
    # ======= II. Iterate through the differentiated time series =======
    for index in returns_series.index[1:]:
        # II.1 Update the cumulative sums
        upward_cumsum = max(0, (1 + upward_cumsum) * (1 + returns_series.loc[index]) - 1)
        downward_cumsum = min(0, (1 + downward_cumsum) * (1 + returns_series.loc[index]) - 1)
        
        # II.2 Check if the cumulative sums exceed the threshold value
        if downward_cumsum < -threshold:
            downward_cumsum = 0
            events_list.append(index)
            
        elif upward_cumsum > threshold:
            upward_cumsum = 0
            events_list.append(index)
            
    # ======= III. Associate the events with the time series =======
    indexed_events = pd.DatetimeIndex(events_list)
    
    return indexed_events

def triple_barrier_labelling(price_series: pd.Series, upper_barrier: float,  lower_barrier: float,  vertical_barrier: int,  volatility_function: str):
    """
    This function labels the events in a price series using the Triple Barrier Method.
    
    Args:
        price_series (pd.Series): Price series of the asset
        upper_barrier (float): Upper barrier threshold
        lower_barrier (float): Lower barrier threshold
        vertical_barrier (int): Vertical barrier window
        volatility_function (str): Function to compute volatility
    
    Returns:
        labeled_series (pd.Series): Labeled series of the events
    """   
    # ======= 0. Auxiliary functions =======
    def observed_volatility(price_series: pd.Series, window: int):
        """
        Computes rolling window volatility using percentage returns.
        
        Args:
            price_series (pd.Series): Price series of the asset
            window (int): Window for the rolling computation
            
        Returns:
            volatility_series (pd.Series): Rolling volatility series
        """
        returns_series = price_series.pct_change().fillna(0)
        volatility_series = returns_series.rolling(window).std() * np.sqrt(window)
        
        return volatility_series
    
    # ======= I. Compute volatility target =======
    if volatility_function == 'observed':
        volatility_series = observed_volatility(price_series=price_series, window=vertical_barrier)

    # ======= II. Initialize the labeled series and trade side =======
    labeled_series = pd.Series(index=price_series.index, dtype=int)
    trade_side = 0 
    
    # ======= III. Iterate through the price series =======
    for index in price_series.index:
        # III.1 Extract the future prices over the horizon
        start_idx = price_series.index.get_loc(index)
        end_idx = min(start_idx + vertical_barrier, len(price_series)) 
        future_prices = price_series.iloc[start_idx:end_idx]

        # III.2 Compute the range of future returns over the horizon
        max_price = future_prices.max()
        min_price = future_prices.min()

        max_price_index = future_prices.idxmax()
        min_price_index = future_prices.idxmin()
        
        max_return = (max_price - price_series.loc[index]) / price_series.loc[index]
        min_return = (min_price - price_series.loc[index]) / price_series.loc[index]

        # III.3 Adjust the barrier thresholds with the volatility
        upper_threshold = upper_barrier * volatility_series.loc[index]
        lower_threshold = lower_barrier * volatility_series.loc[index]

        # III.4 Check if the horiazontal barriers have been hit
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
        
        else: # No position held
            if max_return > upper_threshold:
                long_event = True
            elif min_return < -upper_threshold:
                short_event = True
        
        # III.5 Label the events base on the first event that occurs
        if long_event and short_event: # If both events occur, choose the first one
            if max_price_index < min_price_index:
                labeled_series.loc[index] = 1
            else:
                labeled_series.loc[index] = -1  
                
        elif long_event and not short_event: # If only long event occurs
            labeled_series.loc[index] = 1
            
        elif short_event and not long_event: # If only short event occurs
            labeled_series.loc[index] = -1
            
        else: # If no event occurs (vertical hit)
            labeled_series.loc[index] = 0
        
        # III.6 Update the trade side 
        trade_side = labeled_series.loc[index]

    return labeled_series

