from ..Models import linearRegression as reg

import os

os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import warnings


#! ==================================================================================== #
#! ======================== Series Correlation Functions ============================== #
def get_pearson_correlation(series_1: pd.Series, series_2: pd.Series):
    # ======== I. Compute the correlation ========
    std_a = series_1.std()
    std_b = series_2.std()
    covariance = np.cov(series_1, series_2)[0, 1]

    correlation = covariance / (std_a * std_b)

    return correlation

#*____________________________________________________________________________________ #
def get_distance_correlation(series_1: pd.Series, series_2: pd.Series, distance_measure: str = "manhattan", p: int = 2):
    # ======== I. Define the distance function ========
    distances = {
        "euclidean": get_euclidean_distance,
        "manhattan": get_manhattan_distance,
        "chebyshev": get_chebyshev_distance,
        "minkowski": lambda x, y: get_minkowski_distance(x, y, p),
        "hamming": get_hamming_distance,
        "angular": get_angular_distance,
        "jaccard": get_jaccard_distance,
    }
    if distance_measure not in distances:
        raise ValueError("Unsupported distance measure. Choose from 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'cosine', 'jaccard'.")

    distance_func = distances[distance_measure]

    # ======== II. Compute the distance correlation ========
    n = len(series_1)
    distance_matrix_a = np.zeros((n, n))
    distance_matrix_b = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance_matrix_a[i, j] = distance_func(series_1.iloc[i], series_1.iloc[j])
            distance_matrix_b[i, j] = distance_func(series_2.iloc[i], series_2.iloc[j])

    # ======== III. Center the distance matrices ========
    a = distance_matrix_a - distance_matrix_a.mean(axis=0) - distance_matrix_a.mean(axis=1)[:, np.newaxis] + distance_matrix_a.mean()
    b = distance_matrix_b - distance_matrix_b.mean(axis=0) - distance_matrix_b.mean(axis=1)[:, np.newaxis] + distance_matrix_b.mean()

    # ======== IV. Compute the distance covariance ========
    dCovXY_2 = (a * b).mean()
    dVarXX_2 = (a * a).mean()
    dVarYY_2 = (b * b).mean()

    # ======== V. Compute the distance correlation ========
    dCor = np.sqrt(dCovXY_2 / np.sqrt(dVarXX_2 * dVarYY_2))

    return dCor

#*____________________________________________________________________________________ #
def get_cointegration(series_1: pd.Series, series_2: pd.Series):
    # ======== I. Perform a Linear Regression ========
    model = reg.OLSRegression()
    model.fit(series_2, series_1)

    # ======== II. Extract Regression Coefficients ========
    beta = model.coefficients[0]
    intercept = model.intercept

    # ======== III. Compute Residuals ========
    residuals = series_1 - (beta * series_2 + intercept)

    # ======== IV. Perform ADF & KPSS Tests ========
    adf_results = adfuller(residuals)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_results = kpss(residuals, regression="c", nlags="auto")

    return beta, intercept, adf_results, kpss_results, residuals



#! ==================================================================================== #
#! ========================== Series Distance Functions =============================== #
def get_euclidean_distance(series_1: pd.Series, series_2: pd.Series):
    # ======== I. Compute the Euclidean distance ========
    distance = np.sqrt(np.sum((series_1 - series_2) ** 2))

    return distance

#*____________________________________________________________________________________ #
def get_manhattan_distance(series_1: pd.Series, series_2: pd.Series):
    # ======== I. Compute the Manhattan distance ========
    distance = np.sum(np.abs(series_1 - series_2))

    return distance

#*____________________________________________________________________________________ #
def get_chebyshev_distance(series_1: pd.Series, series_2: pd.Series):
    # ======== I. Compute the Chebyshev distance ========
    distance = np.max(np.abs(series_1 - series_2))

    return distance

#*____________________________________________________________________________________ #
def get_minkowski_distance(series_1: pd.Series, series_2: pd.Series, p: int):
    # ======== I. Compute the Minkowski distance ========
    distance = np.sum(np.abs(series_1 - series_2) ** p) ** (1 / p)

    return distance

#*____________________________________________________________________________________ #
def get_hamming_distance(series_1: pd.Series, series_2: pd.Series):
    # ======== I. Compute the Hamming distance ========
    distance = np.sum(series_1 != series_2)

    return distance

#*____________________________________________________________________________________ #
def get_jaccard_distance(series_1: pd.Series, series_2: pd.Series):
    # ======== I. Compute the Jaccard distance ========
    intersection = np.sum(series_1 & series_2)
    union = np.sum(series_1 | series_2)

    distance = 1 - intersection / union

    return distance

#*____________________________________________________________________________________ #
def get_angular_distance(series_1: pd.Series, series_2: pd.Series):
    # ======== I. Compute the cosine similarity ========
    norm_a = np.sqrt(np.sum(series_1 ** 2))
    norm_b = np.sqrt(np.sum(series_2 ** 2))

    similarity = np.sum(series_1 * series_2) / (norm_a * norm_b)

    # ======== II. Compute the angular distance ========
    distance = 1 - similarity

    return distance

