import pandas as pd
import numpy as np

#! ==================================================================================== #
#! ================================== Series Filters ================================== #
def moving_average(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the moving average =======
    moving_avg = price_series.rolling(window=window + 1).mean()

    # ======= II. Convert to pd.Series and Normalize =======
    moving_avg = pd.Series(moving_avg, index=price_series.index) / (price_series + 1e-8)

    return moving_avg

#*____________________________________________________________________________________ #
def exponential_weightedMA(price_series: pd.Series, window: int, ind_lambda: float):
    # ======= I. Create the weights using a truncated exponential function =======
    weight_range = [(1 - ind_lambda) ** (i - 1) for i in range(1, window + 1)]
    weight_range.reverse()
    weight_range = np.array(weight_range)

    # ======= II. Perform the weighted moving average =======
    series = np.array(price_series)
    wma = get_weightedMA(series=series, weight_range=weight_range)

    # ======= III. Convert to pd.Series =======
    wma = pd.Series(wma, index=price_series.index)

    return wma



#! ==================================================================================== #
#! ================================ Helper functions ================================== #
def get_weightedMA(series: pd.Series, weight_range: np.array):
        # ======= I. Check if the weights are valid =======
        values = np.array(series)
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

#*____________________________________________________________________________________ #
def get_volatility(price_series: pd.Series, window: int):

    returns_series = price_series.pct_change().fillna(0)
    volatility_series = returns_series.rolling(window).std() * np.sqrt(window)

    return volatility_series



#! ==================================================================================== #
#! ================================ Labelling Process ================================= #
def trend_filter(label_series: pd.Series, window: int):
    # ======= I. Create an auxiliary DataFrame =======
    auxiliary_df = pd.DataFrame()
    auxiliary_df["label"] = label_series
    
    # ======= II. Create a group for each label and extract size =======
    auxiliary_df["group"] = (auxiliary_df["label"] != auxiliary_df["label"].shift()).cumsum()
    group_sizes = auxiliary_df.groupby("group")["label"].transform("size")

    # ======= III. Filter the labels based on the group size =======
    auxiliary_df["label"] = auxiliary_df.apply(lambda row: row["label"] if group_sizes[row.name] >= window else 0, axis=1)
    labels_series = auxiliary_df["label"]
    
    return labels_series

