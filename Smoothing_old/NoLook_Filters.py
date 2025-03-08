"""
This file contains multiple filtering methods that can be used to smooth out price series data.

The methods include:

    - Moving Average : Smooth the series by taking the average of the last n values.
    - Exponential Weighted Moving Average : Smooth the series by taking a weighted average of the last n values, with more weight given to the latest values (following an exponential function).

"""

import pandas as pd
import numpy as np


# =======================================================================
# ========================== Filtering Methods ==========================
def moving_average(
    price_series: pd.Series,
    window: int,
):
    """
    This function computes the moving average of a price series.

    Args:
        price_series (pd.Series): The price series of the asset.
        window (int): The window size for the moving average.

    Returns:
        pd.Series: The moving average of the price series.
    """
    # ======= I. Compute the moving average =======
    moving_avg = price_series.rolling(window=window + 1).mean()

    # ======= II. Convert to pd.Series and Normalize =======
    moving_avg = pd.Series(moving_avg, index=price_series.index) / (price_series + 1e-8)

    return moving_avg


# -----------------------------------------------------------------------------
def exponential_weighted_moving_average(price_series: pd.Series, window: int, ind_lambda: float):
    """
    Perform a weighted moving average on a numpy array using a truncated exponential function. The objective is to give more importance to the latest values in the array.

    Args:
        price_series (pd.Series): The price series of the asset.
        window (int): The window size for the moving average.
        ind_lambda (float): The lambda value for the exponential function.

    Returns:
        pd.Series: The weighted moving average of the price series.
    """

    # ======= 0. Intermediate function =======
    def weighted_moving_average(values: np.array, weight_range: np.array):
        """
        Perform a weighted moving average on a numpy array.

        Args:
            values (np.array): The array of values to be averaged.
            weight_range (np.array): The weights to be used in the average. If an integer is passed, the function will use the last n values to calculate the average.

        Returns:
            wma (np.array): The array of weighted averages.
        """
        # ======= I. Check if the weights are valid =======
        values = values.astype("float64")
        wma = values.copy()

        if isinstance(weight_range, int):
            weights = np.array(range(1, weight_range + 1))
            rolling_window = weight_range
        else:
            weights = weight_range
            rolling_window = len(weight_range)

        # ======= II. Calculate the weighted moving average over a rolling window =======
        for i in range(0, len(values)):
            try:
                wma[i] = values[i - rolling_window + 1 : i + 1].dot(weights) / np.sum(weights)
            except:
                wma[i] = np.nan

        return wma

    # ======= I. Create the weights using a truncated exponential function =======
    weight_range = [(1 - ind_lambda) ** (i - 1) for i in range(1, window + 1)]
    weight_range.reverse()
    weight_range = np.array(weight_range)

    # ======= II. Perform the weighted moving average =======
    values = np.array(price_series)
    wma = weighted_moving_average(values=values, weight_range=weight_range)

    # ======= III. Convert to pd.Series =======
    wma = pd.Series(wma, index=price_series.index)

    return wma
