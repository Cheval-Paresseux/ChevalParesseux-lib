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
