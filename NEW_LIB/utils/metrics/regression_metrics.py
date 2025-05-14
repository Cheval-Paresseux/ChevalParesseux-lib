import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
from typing import Optional



#! ==================================================================================== #
#! =============================== Prediction Accuracy ================================ #
def get_regression_rmse(
    predictions: np.array,
    y_true: np.array
) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) for regression predictions.
    
    Parameters:
        - predictions (np.array): The predicted values.
        - y_true (np.array): The actual target values.
    
    Returns:
        - float: The RMSE value.
    """
    # ======= I. Compute Residuals =======
    residuals = y_true - predictions
    
    # ======= II. Compute RMSE =======
    rmse = np.sqrt(np.mean(residuals**2))
    
    return rmse

#*____________________________________________________________________________________ #
def get_regression_mse(
    predictions: np.array,
    y_true: np.array
) -> float:
    """
    Computes the Mean Squared Error (MSE) for regression predictions.
    
    Parameters:
        - predictions (np.array): The predicted values.
        - y_true (np.array): The actual target values.
    
    Returns:
        - float: The MSE value.
    """
    # ======= I. Compute Residuals =======
    residuals = y_true - predictions
    
    # ======= II. Compute MSE =======
    mse = np.mean(residuals**2)
    
    return mse

#*____________________________________________________________________________________ #
def get_regression_smape(
    predictions: np.array, 
    y_true: np.array
) -> float:
    """
    Computes the Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Parameters:
        - predictions (np.array): The predicted values.
        - y_true (np.array): The actual target values.
    
    Returns:
        - float: The SMAPE value (expressed as a percentage).
    """
    # ======= I. Compute Symmetric Absolute Mean =======
    denominator = (np.abs(y_true) + np.abs(predictions)) / 2
    
    # ======= II. Avoid Division by Zero =======
    nonzero_mask = denominator != 0
    smape = np.zeros_like(denominator)
    
    # ======= III. Compute SMAPE =======
    smape[nonzero_mask] = np.abs(predictions[nonzero_mask] - y_true[nonzero_mask]) / denominator[nonzero_mask]
    smape_value = np.mean(smape)
    
    return smape_value

#*____________________________________________________________________________________ #
def get_regression_max_error(
    predictions: np.array, 
    y_true: np.array
) -> float:
    """
    Computes the Maximum Error (Max Error) for regression predictions.
    
    Parameters:
        - predictions (np.array): The predicted values.
        - y_true (np.array): The actual target values.
    
    Returns:
        - float: The Max Error value.
    """
    # ======= I. Compute Absolute Errors =======
    absolute_errors = np.abs(predictions - y_true)
    
    # ======= II. Compute Max Error =======
    max_error = np.max(absolute_errors)
    
    return max_error



#! ==================================================================================== #
#! ============================== Significance Measures =============================== #
def get_regression_r2(
    predictions: np.array, 
    y_true: np.array
) -> float:
    """
    Computes the R-squared value for regression predictions.
    
    Parameters:
        - predictions (np.array): The predicted values.
        - y_train (np.array): The actual target values.
    
    Returns:
        - float: The R-squared value.
    """    
    # ======= I. Total Sum of Squares =======
    SST = np.sum((y_true - np.mean(y_true))**2)
    
    # ======= II. Residual Sum of Squares =======
    SSR = np.sum((y_true - predictions)**2)
    
    # ======= III. R-squared =======
    r2 = 1 - SSR / SST if SST != 0 else 0
    
    return r2

#*____________________________________________________________________________________ #
def get_regression_significance(
    predictions: np.array,
    features_matrix: np.array,
    y_true: np.array,
    coefficients: np.array,
    feature_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Computes t-statistics and p-values for regression coefficients.

    Parameters:
        - predictions (np.array): The predicted values.
        - features_matrix (np.array): The training feature matrix.
        - y_true (np.array): The actual target values.
        - coefficients (np.array): The regression coefficients.
        - feature_names (list[str], optional): Names of the features.

    Returns:
        - pd.DataFrame: A DataFrame containing coefficients, t-stats and p-values.
    """

    # ======= I. Sanity Checks =======
    if np.isnan(features_matrix).any() or np.isnan(y_true).any() or np.isnan(predictions).any():
        raise ValueError("Inputs contain NaNs.")

    if features_matrix.shape[0] != y_true.shape[0] or y_true.shape[0] != predictions.shape[0]:
        raise ValueError("Mismatch in number of observations between X, y, and predictions.")

    nb_observations, nb_features = features_matrix.shape

    if nb_observations <= nb_features:
        raise ValueError("Number of observations must be greater than number of features.")

    # ======= II. Compute Residuals and Variance =======
    residuals = y_true - predictions
    residual_variance = np.sum(residuals**2) / (nb_observations - nb_features)

    # ======= III. Variance-Covariance Matrix =======
    XTX = features_matrix.T @ features_matrix
    var_covar_matrix = residual_variance * np.linalg.pinv(XTX)
    se_coefficients = np.sqrt(np.diag(var_covar_matrix))

    # ======= IV. t-Statistics and p-Values =======
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stats = np.where(se_coefficients != 0, coefficients / se_coefficients, 0)

    degrees_freedom = nb_observations - nb_features
    p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_freedom)) for t_stat in t_stats]

    # ======= V. DataFrame Construction =======
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(nb_features)]

    stats_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "t_stat": t_stats,
        "p_value": p_values
    })

    return stats_df



#! ==================================================================================== #
#! ================================ Residuals Measures ================================ #
def get_durbin_watson(
    residuals: np.array
) -> float:
    """
    Computes the Durbin-Watson statistic for detecting autocorrelation in residuals.
    
    Parameters:
        - residuals (np.array): The residuals of the regression model.
    
    Returns:
        - float: The Durbin-Watson statistic.
    """
    # ======= I. First differences of residuals =======
    residuals_diff = np.diff(residuals)  
    
    # ======= II. Compute Durbin-Watson statistic =======
    dw_stat = np.sum(residuals_diff**2) / np.sum(residuals**2)
    
    return dw_stat

#*____________________________________________________________________________________ #
def get_jarque_bera(
    residuals: np.array
) -> tuple:
    """
    Performs the Jarque-Bera test for normality of residuals.
    
    Parameters:
        - residuals (np.array): The residuals of the regression model.
    
    Returns:
        - tuple: (JB statistic, p-value)
    """
    # ======= I. Compute Skewness and Kurtosis =======
    residuals_series = pd.Series(residuals)
    skewness = residuals_series.skew()
    kurtosis = residuals_series.kurtosis()
    
    # ======= II. Compute Jarque-Bera statistic =======
    n = len(residuals)
    JB_stat = (n / 6) * (skewness ** 2 + (kurtosis ** 2) / 4)
    
    # ===== III. Compute p-value ======= 
    p_value = 1 - (1 / (1 + 0.5 * JB_stat)) ** n
    
    return JB_stat, p_value

#*____________________________________________________________________________________ #
def breusch_pagan_test(
    features_matrix: np.array, 
    residuals: np.array
) -> tuple:
    """
    Performs the Breusch-Pagan test for heteroscedasticity.
    
    Parameters:
        - X (np.array): The training feature matrix (with intercept).
        - residuals (np.array): The residuals of the regression model.
    
    Returns:
        - tuple: (LM statistic, p-value)
    """
    # Step 1: Regress squared residuals on the original X matrix
    residuals_squared = residuals ** 2
    X_with_intercept = np.c_[np.ones(features_matrix.shape[0]), features_matrix]  # Add intercept term
    
    # Step 2: Calculate the coefficient estimates using normal equations
    XTX = np.dot(X_with_intercept.T, X_with_intercept)  # X'X
    XTY = np.dot(X_with_intercept.T, residuals_squared)  # X'Y
    beta_hat = np.linalg.inv(XTX).dot(XTY)  # (X'X)^-1 * X'Y
    
    # Step 3: Calculate the residuals of the regression on squared residuals
    residuals_squared_pred = np.dot(X_with_intercept, beta_hat)
    residuals_squared_error = residuals_squared - residuals_squared_pred
    
    # Step 4: Calculate LM statistic (Breusch-Pagan)
    lm_stat = (np.sum(residuals_squared_error ** 2) / 2) / np.var(residuals_squared_pred)
    
    # Step 5: Compute p-value using chi-square distribution (approximation)
    n = len(residuals)
    p_value = 1 - np.exp(-lm_stat / 2)  # Approximation of the chi-square distribution
    
    return lm_stat, p_value


