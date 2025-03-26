from ..Models import linearRegression as reg

import numpy as np
import pandas as pd


#! ==================================================================================== #
#! ====================== Series Tendency Statistics Functions ======================== #
def get_momentum(series: pd.Series):
    
    first_value = series.iloc[0]
    last_value = series.iloc[-1]
    
    momentum = (last_value - first_value) / first_value
    
    return momentum

#*____________________________________________________________________________________ #
def get_Z_momentum(series: pd.Series):
    # ======= I. Compute Momentum =======
    momentum = get_momentum(series)
    
    # ======= II. Compute Standard Deviation of Returns =======
    returns_series = series.pct_change().dropna()
    returns_standard_deviation = np.std(returns_series)
    
    # ======= III. Compute Z-Momentum =======
    Z_momentum = momentum / returns_standard_deviation
    
    return Z_momentum

#*____________________________________________________________________________________ #
def get_simple_TempReg(series: pd.Series):
    # ======= I. Fit the temporal regression =======
    X = np.arange(len(series))
    model = reg.OLSRegression()
    model.fit(X, series)
    
    # ======= II. Extract the coefficients and statistics =======
    coefficients = model.coefficients
    intercept = model.intercept
    statistics = model.statistics
    residuals = model.residuals

    return intercept, coefficients, statistics,residuals

#*____________________________________________________________________________________ #
def get_quad_TempReg(series: pd.Series):
    # ======= 1. Fit the temporal regression =======
    X = np.arange(len(series))
    X = np.column_stack((X, X**2))
    model = reg.OLSRegression()
    model.fit(X, series)
    
    # ======= 2. Extract the coefficients and statistics =======
    coefficients = model.coefficients
    intercept = model.intercept
    statistics = model.statistics
    residuals = model.residuals
    

    return intercept, coefficients, statistics, residuals

#*____________________________________________________________________________________ #
def get_OU_estimation(series: pd.Series):
    # ======== I. Initialize series ========
    series_array = np.array(series)
    differentiated_series = np.diff(series_array)
    mu = np.mean(series)
    
    X = series_array[:-1] - mu  # X_t - mu
    Y = differentiated_series  # X_{t+1} - X_t

    # ======== II. Perform OLS regression ========
    model = reg.OLSRegression()
    model.fit(Y, X)
    
    # ======== III. Extract Parameters ========
    theta = -model.coefficients[0]
    if theta > 0:
        residuals = model.residuals
        sigma = np.sqrt(np.var(residuals) * 2 * theta)
        half_life = np.log(2) / theta
    else:
        theta = 0
        sigma = 0
        half_life = 0

    return mu, theta, sigma, half_life

#*____________________________________________________________________________________ #  