
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Self, Optional, Iterable



#! ==================================================================================== #
#! ============================== Tree Visualization ================================== #
def plot_tree(
    node,
    depth: int = 0, 
    x: float = 0.5, 
    y: float = 1, 
    dx: float = 0.3, 
    dy: float = 0.2, 
    ax: Optional[plt.Axes] = None, 
    feature_names: Optional[Iterable[str]] = None,
) -> None:
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

#*____________________________________________________________________________________ #
