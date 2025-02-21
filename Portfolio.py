"""
# Description: This file contains the functions used to compute the optimal weights of a portfolio using different optimization methods.
_____
covmat_expectret_returnsdf: Prepares the different dataframes/array to be used in the optimization process.
riskfolio_optimization: Compute the optimal weights of a portfolio using Riskfolio-Lib, the parameteres usually used are Markowitz efficient frontier.
targetVolatility_optimization: This function will compute the optimal weights of a portfolio using a target volatility.
_____
POTENTIAL IMPROVEMENTS:
    - Add more optimization methods.
"""

import pandas as pd
import numpy as np
import riskfolio as rp
import cvxpy
from scipy.linalg import sqrtm

import warnings

warnings.filterwarnings("ignore")


# ========================================================================================================== #
def covmat_expectret_returnsdf(combinations_list: list):
    """
    Prepares the different dataframes/array to be used in the optimization process.

    Args:
        combinations_list (list): List containing the different combinations of assets.

    Returns:
        cov_matrix (pd.DataFrame): Covariance matrix of the returns.
        expected_return_vector (np.array): Expected returns of each combination.
        returns_df (pd.DataFrame): DataFrame containing the returns of each combination.
    """
    # ======== I. Getting the residuals of each combination ========
    expected_return_vector = []
    residuals_dict = {}
    for comb_dict in combinations_list:
        # ---------
        residuals = comb_dict["residuals"]
        zscore = comb_dict["z_score"]
        residuals = residuals

        if zscore < 0:  # This is performed in order to create a long-only portfolio
            residuals = -residuals

        assets = comb_dict["assets_name"]
        expected_return = comb_dict["expected_return"]

        # ---------
        comb_name = "_".join(assets)

        # ---------
        residuals_dict[comb_name] = residuals
        expected_return_vector.append(expected_return)

    # ======== II. Storing the data ========
    residual_dataframe = pd.DataFrame(residuals_dict)

    # --------- Those are exactly the returns of the pair portfolio
    returns_df = residual_dataframe.diff().dropna()

    # ---------
    expected_return_vector = np.array(expected_return_vector)
    cov_matrix = returns_df.cov()

    return cov_matrix, expected_return_vector, returns_df


# ---------------------------------------------------------------------------------------------------------- #
def riskfolio_optimization(
    returns_df: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    expected_return_vector: np.array,
    model: str,
    risk_measure: str,
    objective_function: str,
):
    """
    Compute the optimal weights of a portfolio using Markowitz efficient frontier.

    Args:
        returns_df (pd.DataFrame): DataFrame containing the returns of each combination.
        cov_matrix (pd.DataFrame): Covariance matrix of the returns.
        expected_return_vector (np.array): Expected returns of each combination.
        model (str): Model used for the optimization. Possible inputs : "Classic", "BL" (Black-Litterman) or "FM" (Factor Model).
        risk_measure (str): Risk measure used for the optimization. Possible inputs : "MV" (Mean-Variance), "CDaR" (Conditional Drawdown at Risk), "Volatility" or "MVaR" (Mean-Value at Risk).
        objective_function (str): Objective function used for the optimization. Possible inputs : "MaxRet" (Maximize return), "MinRisk" (Minimize risk), "Utility" or "Sharpe" (Maximize Sharpe ratio).

    Returns:
        weights (pd.DataFrame): DataFrame containing the optimal weights for each combination.
    """
    # ======== I. Initializing with the right inputs ========
    port = rp.Portfolio(returns_df)
    port.mu = expected_return_vector.T
    port.cov = cov_matrix

    # ---------
    hist = True  # Use historical scenarios for risk measures that depend on scenarios
    rf = 0  # Risk free rate
    aversion = 0  # Risk aversion factor, only useful when obj is 'Utility'

    # ======== II. Estimate optimal portfolio ========
    if model == "Classic":
        try:
            w = port.optimization(
                model=model,
                rm=risk_measure,
                obj=objective_function,
                rf=rf,
                l=aversion,
                hist=hist,
            )
            if w is None:
                weights = pd.DataFrame(np.zeros(returns_df.shape[1])).T
            else:
                weights = w.T
            return weights
        except Exception as e:
            pass

    elif model == "RiskParity":
        try:
            w = port.rp_optimization()
            if w is None:
                weights = pd.DataFrame(np.zeros(returns_df.shape[1])).T
            else:
                weights = w.T
            return weights
        except Exception as e:
            pass

    weights = pd.DataFrame(np.zeros(returns_df.shape[1])).T
    return weights


# ---------------------------------------------------------------------------------------------------------- #
def targetVolatility_optimization(
    exp_ret: np.array,
    cov: np.array,
    boundary: np.array,
    target_volatility: float,
):
    """
    This function will compute the optimal weights of a portfolio using a target volatility.
    If the first attempt fails, it will increase the target volatility and retry once.

    Args:
        exp_ret (np.array): Expected returns of each combination.
        cov (np.array): Covariance matrix of the returns.
        boundary (np.array): Lower and upper bounds for each asset.
        vol_target (float): Target volatility.

    Returns:
        opt_x (pd.DataFrame): A DataFrame containing the optimal weights for each combination.
    """
    # ======== I. Setting up the optimization problem ========
    N = len(exp_ret)
    cov += np.eye(cov.shape[0]) * 1e-6  # Regularize if necessary

    # -------- Decision variable
    x = cvxpy.Variable(N)

    # -------- Constant vectors and matrices
    mu = cvxpy.Constant(exp_ret)
    G = cvxpy.Constant(np.real(sqrtm(cov)))

    # -------- Objective: Maximize expected returns
    obj = cvxpy.Maximize(x @ mu)

    # -------- Constraints
    def setup_constraints(vol_target):
        cons = [cvxpy.norm(G @ x, 2) <= vol_target]  # Volatility constraint
        for i in range(N):
            cons.append(boundary[i, 0] <= x[i])  # Lower bound
            cons.append(x[i] <= boundary[i, 1])  # Upper bound
        cons.append(cvxpy.sum(x) == 1)
        return cons

    # ======= II. Attempt to solve the problem ========
    for attempt in range(2):  # Try up to two attempts
        cons = setup_constraints(target_volatility)
        prob = cvxpy.Problem(obj, cons)

        try:
            prob.solve(ignore_dpp=True, verbose=False)
            opt_x = x.value
            if opt_x is not None:
                weights = pd.DataFrame(opt_x).T
                return weights  # Return solution if successful
        except Exception as e:
            pass

        # If the first attempt fails, increase vol_target
        if attempt == 0:
            target_volatility += 0.1 / np.sqrt(252)

    # Return zeros if both attempts fail
    weights = pd.DataFrame(np.zeros(N)).T
    return weights


# ---------------------------------------------------------------------------------------------------------- #
def equalWeights_optimization(
    returns_df: pd.DataFrame,
):
    """
    Compute the optimal weights of a portfolio using equal weights.

    Args:
        returns_df (pd.DataFrame): DataFrame containing the returns of each combination.

    Returns:
        weights (pd.DataFrame): DataFrame containing the optimal weights for each combination
    """
    nb_assets = returns_df.shape[1]
    weights = pd.DataFrame(np.ones(nb_assets) / nb_assets).T

    return weights


# ---------------------------------------------------------------------------------------------------------- #
def inverseVolatility_optimization(cov_matrix: pd.DataFrame):
    """
    Compute the optimal weights of a portfolio using the inverse of the volatility.

    Args:
        cov_matrix (pd.DataFrame): Covariance matrix of the returns.

    Returns:
        weights (pd.DataFrame): DataFrame containing the optimal weights for each asset.
    """
    # ======== I. Computing the volatility of each asset ========
    volatilities = np.sqrt(np.diag(cov_matrix))

    # ======== II. Computing the inverse volatility weights ========
    inverse_vol = 1 / volatilities
    weights = inverse_vol / inverse_vol.sum()

    # ======== III. Formatting weights as a DataFrame ========
    weights = pd.DataFrame(weights).T

    return weights
