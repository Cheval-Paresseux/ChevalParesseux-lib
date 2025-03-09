import numpy as np
import pandas as pd
import statsmodels.api as sm


# ==================================================================================== #
# ===================== Series Descriptive Statistics Functions ====================== #
def get_minimum(series: pd.Series):
    minimum = np.min(series)
    
    return minimum
# ____________________________________________________________________________________ #
def get_maximum(series: pd.Series):
    maximum = np.max(series)
    
    return maximum 
# ____________________________________________________________________________________ #
def get_average(series: pd.Series):
    average = np.mean(series)
    
    return average
# ____________________________________________________________________________________ #
def get_median(series: pd.Series):
    median = np.median(series)
    
    return median
# ____________________________________________________________________________________ #
def get_standard_deviation(series: pd.Series):
    standard_deviation = np.std(series)
    
    return standard_deviation
# ____________________________________________________________________________________ #
def get_variance(series: pd.Series):
    variance = np.var(series)
    
    return variance
# ____________________________________________________________________________________ #
def get_skewness(series: pd.Series):
    skewness = series.skew()
    
    return skewness
# ____________________________________________________________________________________ #
def get_kurtosis(series: pd.Series):
    kurtosis = series.kurtosis()
    
    return kurtosis

# ==================================================================================== #
# ======================= Series Tendency Statistics Functions ======================= #
def get_momentum(series: pd.Series):
    first_value = series.iloc[0]
    last_value = series.iloc[-1]
    
    momentum = (last_value - first_value) / first_value
    
    return momentum
# ____________________________________________________________________________________ #
def get_Z_momentum(series: pd.Series):
    first_value = series.iloc[0]
    last_value = series.iloc[-1]
    returns_series = series.pct_change().dropna()
    returns_standard_deviation = np.std(returns_series)
    
    momentum = (last_value - first_value) / first_value
    Z_momentum = momentum / returns_standard_deviation
    
    return Z_momentum
# ____________________________________________________________________________________ #
def get_simpleOLS(series: pd.Series):
    # ------- 1. Fit the OLS regression -------
    X = np.arange(len(series))
    X = sm.add_constant(X)  # Add intercept for OLS
    model = sm.OLS(series, X, missing="drop")
    results = model.fit()

    # ------- 2. Extract the trend coefficient and t-statistic -------
    trend_coefficient = results.params[1]
    t_statistic = results.tvalues[1]

    return trend_coefficient, t_statistic
# ____________________________________________________________________________________ #
def get_quadOLS(series: pd.Series):
    # ------- 1. Fit the OLS regression -------
    X = np.arange(len(series))
    X_quad = np.column_stack((X, X**2))
    X_quad = sm.add_constant(X_quad)
    model = sm.OLS(series, X_quad, missing="drop")
    results = model.fit()

    # ------- 2. Extract the trend coefficient, acceleration coefficient, and t-statistic -------
    trend_coefficient = results.params[1]
    acceleration_coefficient = results.params[2]
    t_statistic = results.tvalues[2]

    return trend_coefficient, acceleration_coefficient, t_statistic