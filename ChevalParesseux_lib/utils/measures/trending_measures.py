from ..tools import regression_models as reg

import numpy as np
import pandas as pd



#! ==================================================================================== #
#! ====================== Series Tendency Statistics Functions ======================== #
def get_momentum(
    series: pd.Series
) -> float:
    """
    Compute the momentum of a series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - momentum (float): Momentum value.
    """
    # ======= I. Extract first and last value =======
    first_value = series.iloc[0]
    last_value = series.iloc[-1]
    
    # ======= II. Compute Momentum =======
    momentum = (last_value - first_value) / first_value
    
    return momentum

#*____________________________________________________________________________________ #
def get_Z_momentum(
    series: pd.Series
) -> float:
    """
    Compute the Z-momentum of a series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - Z_momentum (float): Z-momentum value.
    """
    # ======= I. Compute Momentum =======
    momentum = get_momentum(series)
    
    # ======= II. Compute Standard Deviation of Returns =======
    returns_series = series.pct_change().dropna()
    returns_standard_deviation = np.std(returns_series)
    
    # ======= III. Compute Z-Momentum =======
    Z_momentum = momentum / returns_standard_deviation
    
    return Z_momentum

#*____________________________________________________________________________________ #
def get_simple_TempReg(
    series: pd.Series
) -> tuple:
    """
    Compute the simple temporal regression of a series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - intercept (float): Intercept of the regression.
        - coefficients (np.ndarray): Coefficients of the regression.
        - statistics (dict): Statistics of the regression.
        - residuals (np.ndarray): Residuals of the regression.
    """
    # ======= I. Fit the temporal regression =======
    X = np.arange(len(series))
    model = reg.OLS_regression()
    model.fit(X, series)
    
    # ======= II. Extract the coefficients and statistics =======
    coefficients = model.coefficients
    intercept = model.intercept
    metrics = model.metrics

    return intercept, coefficients, metrics

#*____________________________________________________________________________________ #
def get_quad_TempReg(
    series: pd.Series
) -> tuple:
    """
    Compute the quadratic temporal regression of a series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - intercept (float): Intercept of the regression.
        - coefficients (np.ndarray): Coefficients of the regression.
        - statistics (dict): Statistics of the regression.
        - residuals (np.ndarray): Residuals of the regression.
    """
    # ======= 1. Fit the temporal regression =======
    X = np.arange(len(series))
    X = np.column_stack((X, X**2))
    model = reg.OLS_regression()
    model.fit(X, series)
    
    # ======= 2. Extract the coefficients and statistics =======
    coefficients = model.coefficients
    intercept = model.intercept
    metrics = model.metrics
    
    return intercept, coefficients, metrics

#*____________________________________________________________________________________ #
def get_OU_estimation(
    series: pd.Series
) -> tuple:
    """
    Compute the Ornstein-Uhlenbeck estimation of a series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - mu (float): Mean of the series.
        - theta (float): Speed of mean reversion.
        - sigma (float): Volatility of the series.
        - half_life (float): Half-life of the series.
    """
    # ======== I. Initialize series ========
    series_array = np.array(series)
    differentiated_series = np.diff(series_array)
    mu = np.mean(series)
    
    X = series_array[:-1] - mu  # X_t - mu
    y = differentiated_series  # X_{t+1} - X_t

    # ======== II. Perform OLS regression ========
    model = reg.OLS_regression()
    model.fit(X, y)
    
    # ======== III. Extract Parameters ========
    theta = -model.coefficients[0]
    if theta > 0:
        residuals = y - (model.intercept + np.dot(X, model.coefficients))
        sigma = np.sqrt(np.var(residuals) * 2 * theta)
        half_life = np.log(2) / theta
    else:
        theta = 0
        sigma = 0
        half_life = 0

    return mu, theta, sigma, half_life

#*____________________________________________________________________________________ #
def get_autocorrelation(
    series: pd.Series, 
    lags: int
) -> np.ndarray:
    """
    Compute autocorrelations up to given number of lags.
    
    Parameters:
        - series (pd.Series): Time series data.
        - lags (int): Number of lags to compute.
    
    Returns:
        - np.ndarray: Autocorrelation values for each lag.
    """
    # ======= I. Standardize the series =======
    standardized_series = (series - series.mean()) / series.std()
    
    # ======= II. Compute autocorrelations =======
    result = [] 
    for lag in range(1, lags + 1):
        acf_lag = np.corrcoef(standardized_series[lag:], standardized_series[:-lag])[0, 1]
        result.append(acf_lag)
    
    # ======= III. Transform into np.array =======
    result = np.array(result)

    return result

#*____________________________________________________________________________________ #
def get_partial_autocorrelation(
    series: pd.Series, 
    lags: int
) -> np.ndarray:
    """
    Compute partial autocorrelation using linear regression (OLS) method.
    
    Parameters:
        - series (pd.Series): Time series data.
        - lags (int): Number of lags to compute.
    
    Returns:
        - np.ndarray: Partial autocorrelation values for each lag.
    """
    # ======= I. Iterate through lags =======
    pacf = []  
    for lag in range(1, lags + 1):
        # ----- 1. Create lagged DataFrame -----
        df = pd.DataFrame({'y': series[lag:]})
        for i in range(1, lag + 1):
            df[f'lag_{i}'] = series.shift(i)[lag:]

        # ----- 2. Extract explanatory and target series -----
        df = df.dropna()
        X = df.drop(columns='y').values
        y = df['y'].values

        # ----- 3. Fit OLS regression -----
        reg_model = reg.OLS_regression()
        reg_model.fit(X, y)
        last_coeff = reg_model.coefficients[-1] # Last coefficient is PACF at lag k
        pacf.append(last_coeff)  
    
    # ======= II. Transform into np.array =======
    pacf = np.array(pacf)

    return pacf

