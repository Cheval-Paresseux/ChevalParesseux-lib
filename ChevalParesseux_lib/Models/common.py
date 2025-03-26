import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import t

#! ==================================================================================== #
#! =================================== Base Models ==================================== #
class ML_Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self):
        pass

    #?____________________________________________________________________________________ #
    @abstractmethod
    def fit(self):
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def predict(self):
        pass

#! ==================================================================================== #
#! ================================= Helper Functions ================================= #
def adapt_learning_rate(learning_rate: float, loss: float, last_loss: float):
    new_rate = learning_rate
    if loss > last_loss:
        new_rate /= 2
    else:
        new_rate *= 1.05
    
    return new_rate

#*____________________________________________________________________________________ #
def early_stopping(loss: float, last_loss: float):
    # ======= I. Check the loss diference =======
    if last_loss == np.inf:
        return False
    
    loss_diff = np.abs(loss - last_loss)
    early_stop = False
    
    # ======= II. Check if the loss difference is small enough =======
    if loss_diff < 1e-5:
        early_stop = True
    
    return early_stop

#*____________________________________________________________________________________ #
def get_regression_stats(predictions: np.array, X_train: np.array, y_train: np.array, coefficients: np.array):
    """
    Computes regression statistics including R-squared, variance, and p-values.
    """
    # ======= I. Compute Residuals =======
    residuals = y_train - predictions
    
    # ======= II. Compute Residual Statistics =======
    nb_observations, nb_features = X_train.shape

    if nb_observations <= nb_features:
        raise ValueError("Number of observations must be greater than the number of features to compute statistics.")

    variance = np.sum(residuals**2) / (nb_observations - nb_features)
    mean = np.mean(residuals)
    median = np.median(residuals)

    # ======= III. Compute R-Squared =======
    SST = np.sum((y_train - np.mean(y_train))**2)
    SSR = np.sum((predictions - np.mean(y_train))**2)
    R_squared = SSR / SST
    
    # ======= IV. Compute t-Statistics and p-Values =======
    XTX = X_train.T @ X_train

    # Use pseudo-inverse to avoid singularity issues
    var_covar_matrix = variance * np.linalg.pinv(XTX)
    se_coefficients = np.sqrt(np.diag(var_covar_matrix))
    t_stats = coefficients / se_coefficients

    # Degrees of freedom check
    degrees_freedom = nb_observations - nb_features
    p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_freedom)) for t_stat in t_stats]

    # ======= V. Store the Statistics =======
    statistics = {
        "Variance": variance,
        "Mean": mean,
        "Median": median,
        "R_squared": R_squared,
        "T_stats": t_stats.tolist(),
        "P_values": p_values
    }

    return statistics, residuals