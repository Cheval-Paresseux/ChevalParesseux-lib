import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import t
import matplotlib.pyplot as plt

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

#*____________________________________________________________________________________ #
def plot_tree(node, depth=0, x=0.5, y=1.0, dx=0.3, dy=0.2, ax=None, feature_names=None):
    """
    Recursively plots a decision tree using Matplotlib.

    Parameters:
    - node: Root node of the tree (Node class)
    - depth: Current depth of recursion
    - x, y: Position of the current node
    - dx, dy: Horizontal & vertical spacing
    - ax: Matplotlib axis (created if None)
    - feature_names: List of feature names (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    if node.is_leaf_node():  
        label = f"Leaf\nClass: {node.value}\nSamples: {node.samples}\nImpurity: {node.impurity:.2f}"
        ax.text(x, y, label, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightblue"))
    
    else:  
        feature_label = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
        label = f"{feature_label} ≤ {node.threshold:.2f}\nSamples: {node.samples}\nImpurity: {node.impurity:.2f}"
        ax.text(x, y, label, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

        # Child positions
        xl, xr = x - dx / (2 ** depth), x + dx / (2 ** depth)
        yl, yr = y - dy, y - dy

        # Draw edges
        ax.plot([x, xl], [y - 0.02, yl + 0.02], "k-", lw=1)
        ax.plot([x, xr], [y - 0.02, yr + 0.02], "k-", lw=1)

        # Recursively plot children
        plot_tree(node.left, depth + 1, xl, yl, dx, dy, ax, feature_names)
        plot_tree(node.right, depth + 1, xr, yr, dx, dy, ax, feature_names)

    if ax is None:
        plt.show()
    
    return None