import auxiliary as aux

import pandas as pd
import numpy as np


# =======================================================================
# ========================== Filtering Methods ==========================
def moving_average(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the moving average =======
    moving_avg = price_series.rolling(window=window + 1).mean()

    # ======= II. Convert to pd.Series and Normalize =======
    moving_avg = pd.Series(moving_avg, index=price_series.index) / (price_series + 1e-8)

    return moving_avg


# -----------------------------------------------------------------------------
def exponential_weightedMA(price_series: pd.Series, window: int, ind_lambda: float):
    # ======= I. Create the weights using a truncated exponential function =======
    weight_range = [(1 - ind_lambda) ** (i - 1) for i in range(1, window + 1)]
    weight_range.reverse()
    weight_range = np.array(weight_range)

    # ======= II. Perform the weighted moving average =======
    values = np.array(price_series)
    wma = aux.get_weightedMA(values=values, weight_range=weight_range)

    # ======= III. Convert to pd.Series =======
    wma = pd.Series(wma, index=price_series.index)

    return wma
