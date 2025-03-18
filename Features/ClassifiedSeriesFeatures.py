import sys
sys.path.append("../")
from Features import auxiliary as aux

import pandas as pd
import numpy as np

#! ==================================================================================== #
#! ============================= Evolution Measure Features =========================== #
def average_features(
    predictions_series: pd.Series,
    window: int,
):
    # ======= I. Compute the rolling average =======
    rolling_avg = predictions_series.rolling(window=window + 1).mean()

    # ======= II. Convert to pd.Series and Normalize =======
    rolling_avg = pd.Series(rolling_avg, index=predictions_series.index)
    
    # ======= III. Change Name =======
    rolling_avg.name = f"average_{window}"

    return rolling_avg

#*____________________________________________________________________________________ #
def volatility_features(
    predictions_series: pd.Series,
    window: int,
):
    # ======= I. Compute the rolling volatility =======
    rolling_volatility = predictions_series.rolling(window=window + 1).std()

    # ======= II. Convert to pd.Series and Normalize =======
    rolling_volatility = pd.Series(rolling_volatility, index=predictions_series.index)
    
    # ======= III. Change Name =======
    rolling_volatility.name = f"volatility_{window}"

    return rolling_volatility

#*____________________________________________________________________________________ #
def changes_features(
    predictions_series: pd.Series,
    window: int,
):
    # ======= 0. Auxiliary function =======
    def compute_changes(series: pd.Series):
        diff_series = series.diff() ** 2
        changes_count = diff_series[diff_series > 0].count()

        return changes_count

    # ======= I. Compute the rolling changes =======
    rolling_changes = predictions_series.rolling(window=window + 1).apply(compute_changes, raw=False)

    # ======= II. Convert to pd.Series and Normalize =======
    rolling_changes = pd.Series(rolling_changes, index=predictions_series.index)
    
    # ======= III. Change Name =======
    rolling_changes.name = f"changes_{window}"

    return rolling_changes

#*____________________________________________________________________________________ #
def entropy_features(
    predictions_series: pd.Series,
    window: int,
):
    # ======= I. Compute the rolling entropy features =======
    rolling_shannon = predictions_series.rolling(window=window + 1).apply(aux.get_shannon_entropy, raw=False)
    rolling_plugin = predictions_series.rolling(window=window + 1).apply(aux.get_plugin_entropy, raw=False)
    rolling_lempel_ziv = predictions_series.rolling(window=window + 1).apply(aux.get_lempel_ziv_entropy, raw=False)
    rolling_kontoyiannis = predictions_series.rolling(window=window + 1).apply(aux.get_kontoyiannis_entropy, raw=False)

    # ======= II. Convert to pd.Series =======
    rolling_shannon = pd.Series(rolling_shannon, index=predictions_series.index)
    rolling_plugin = pd.Series(rolling_plugin, index=predictions_series.index)
    rolling_lempel_ziv = pd.Series(rolling_lempel_ziv, index=predictions_series.index)
    rolling_kontoyiannis = pd.Series(rolling_kontoyiannis, index=predictions_series.index)
    
    # ======= III. Change Names =======
    rolling_shannon.name = f"shannon_entropy_{window}"
    rolling_plugin.name = f"plugin_entropy_{window}"
    rolling_lempel_ziv.name = f"lempel_ziv_entropy_{window}"
    rolling_kontoyiannis.name = f"kontoyiannis_entropy_{window}"

    return rolling_shannon, rolling_plugin, rolling_lempel_ziv, rolling_kontoyiannis

