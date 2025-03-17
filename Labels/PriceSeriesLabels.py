import sys
sys.path.append("../")
from Models import LinearRegression as reg
from Labels import auxiliary as aux

import numpy as np
import pandas as pd


# ==================================================================================== #
# =================================== LABELLERS ====================================== #
def tripleBarrier_labeller(price_series: pd.Series, params: dict):
    # ======= 0. Params extraction =======
    upper_barrier = params["upper_barrier"]
    lower_barrier = params["lower_barrier"]
    vertical_barrier = params["vertical_barrier"]

    # ======= I. Compute volatility target =======
    volatility_series = aux.get_volatility(price_series=price_series, window=vertical_barrier)

    # ======= II. Initialize the labeled series and trade side =======
    labeled_series = pd.Series(index=price_series.index, dtype=int)
    trade_side = 0

    # ======= III. Iterate through the price series =======
    for index in price_series.index:
        # III.1 Extract the future prices over the horizon
        start_idx = price_series.index.get_loc(index)
        end_idx = min(start_idx + vertical_barrier, len(price_series))
        future_prices = price_series.iloc[start_idx:end_idx]

        # III.2 Compute the range of future returns over the horizon
        max_price = future_prices.max()
        min_price = future_prices.min()

        max_price_index = future_prices.idxmax()
        min_price_index = future_prices.idxmin()

        max_return = (max_price - price_series.loc[index]) / price_series.loc[index]
        min_return = (min_price - price_series.loc[index]) / price_series.loc[index]

        # III.3 Adjust the barrier thresholds with the volatility
        upper_threshold = upper_barrier * volatility_series.loc[index]
        lower_threshold = lower_barrier * volatility_series.loc[index]

        # III.4 Check if the horizontal barriers have been hit
        long_event = False
        short_event = False

        if trade_side == 1:  # Long trade
            if max_return > upper_threshold:
                long_event = True
            elif min_return < -lower_threshold:
                short_event = True

        elif trade_side == -1:  # Short trade
            if min_return < -upper_threshold:
                short_event = True
            elif max_return > lower_threshold:
                long_event = True

        else:  # No position held
            if max_return > upper_threshold:
                long_event = True
            elif min_return < -upper_threshold:
                short_event = True

        # III.5 Label based on the first event that occurs
        if long_event and short_event:  # If both events occur, choose the first one
            if max_price_index < min_price_index:
                labeled_series.loc[index] = 1
            else:
                labeled_series.loc[index] = -1

        elif long_event and not short_event:  # If only long event occurs
            labeled_series.loc[index] = 1

        elif short_event and not long_event:  # If only short event occurs
            labeled_series.loc[index] = -1

        else:  # If no event occurs (vertical hit)
            labeled_series.loc[index] = 0

        # III.6 Update the trade side
        trade_side = labeled_series.loc[index]

    return labeled_series

# ____________________________________________________________________________________ #
def lookForward_labeller(price_series: pd.Series, params: dict):
    # ======= 0. Params extraction & Initialization =======
    price_series = price_series.dropna()

    size_window_smooth = params["size_window_smooth"]
    lambda_smooth = params["lambda_smooth"]
    trend_size = params["trend_size"]
    volatility_threshold = params["volatility_threshold"]

    auxiliary_df = price_series.to_frame()
    auxiliary_df["smooth_close"] = aux.exponential_weightedMA(price_series, size_window_smooth, lambda_smooth)

    # ======= I. Significant look forward Label =======
    # ------- 1. Get the moving X days returns and the moving X days volatility -------
    auxiliary_df["Xdays_returns"] = (auxiliary_df["smooth_close"].shift(-size_window_smooth) - auxiliary_df["smooth_close"]) / auxiliary_df["smooth_close"]
    auxiliary_df["Xdays_vol"] = auxiliary_df["Xdays_returns"].rolling(window=size_window_smooth).std()

    # ------- 2. Compare the X days returns to the volatility  -------
    auxiliary_df["Xdays_score"] = auxiliary_df["Xdays_returns"] / auxiliary_df["Xdays_vol"]
    auxiliary_df["Xdays_label"] = auxiliary_df["Xdays_score"].apply(lambda x: 1 if x > volatility_threshold else (-1 if x < -volatility_threshold else 0))

    # ------- 3. Eliminate the trends that are too small -------
    label_series = aux.stabilize_labels(label_series=auxiliary_df["Xdays_label"], window=trend_size)

    return label_series

# ____________________________________________________________________________________ #
def regR2rank_labeller(price_series: pd.Series, params: dict):
    # ======= I. Extract the parameters =======
    size_window_smooth = int(params["size_window_smooth"])
    lambda_smooth = params["lambda_smooth"]
    horizon = int(params["horizon"])
    horizon_extension = params["horizon_extension"]
    r2_threshold = params["upper_r2_threshold"]
    trend_size = int(params["trend_size"])

    # ======= II. Initialize the series =======
    ewma_series = aux.exponential_weightedMA(price_series=price_series, window=size_window_smooth, ind_lambda=lambda_smooth)
    nb_elements = len(ewma_series)

    labeled_series = pd.Series(0, index=price_series.index, dtype=int)  # Initialise à 0

    # ======= III. Labelling Process =======
    horizon_max = round(horizon * (1 + horizon_extension))
    for idx in range(nb_elements - horizon + 1):
        # III.0 Skip the NaN values
        if pd.isna(ewma_series.iloc[idx]):  # Correction ici
            continue

        # III.1 Iterate over different horizons to find the most significant trend
        best_r2 = 0
        for current_horizon in range(horizon, horizon_max):
            # ------ 1. Extract the future EMA values ------
            future_ewma = ewma_series.iloc[idx:idx + current_horizon]
            temporality = np.arange(len(future_ewma))  # Correction ici

            # ------ 2. Fit the Linear Regression and Extract R² ------
            model = reg.OLSRegression()
            model.fit(temporality, future_ewma)
            statistics, _ = model.get_statistics()
            r2 = statistics["R_squared"]
            slope = model.coefficients[0]

            # ------ 3. Check if the trend is significant ------
            if r2 > best_r2 and r2 > r2_threshold:
                best_r2 = r2
                labeled_series.iloc[idx] = 1 if slope > 0 else -1  # Correction ici
    
    # ------- 3. Eliminate the trends that are too small -------
    label_series = aux.stabilize_labels(label_series=labeled_series, window=trend_size)

    return label_series

# ____________________________________________________________________________________ #
def boostedLF_labeller(price_series: pd.Series, params: dict):
    # ======= 0. Params extraction & Initialization =======
    price_series = price_series.dropna()

    size_window_smooth = params["size_window_smooth"]
    lambda_smooth = params["lambda_smooth"]
    trend_size = params["trend_size"]
    volatility_threshold = params["volatility_threshold"]

    results_df = price_series.to_frame()
    results_df["smooth_close"] = aux.exponential_weightedMA(price_series, size_window_smooth, lambda_smooth)

    # ======= I. Significant look forward Label =======
    # ------- 1. Get the moving X days returns and the moving X days volatility -------
    results_df["Xdays_returns"] = (results_df["smooth_close"].shift(-size_window_smooth) - results_df["smooth_close"]) / results_df["smooth_close"]
    results_df["Xdays_vol"] = results_df["Xdays_returns"].rolling(window=size_window_smooth).std()

    # ------- 2. Compare the X days returns to the volatility  -------
    results_df["Xdays_score"] = results_df["Xdays_returns"] / results_df["Xdays_vol"]
    results_df["Xdays_label"] = results_df["Xdays_score"].apply(lambda x: 1 if x > volatility_threshold else (-1 if x < -volatility_threshold else 0))

    # ------- 3. Eliminate the trends that are too small -------
    results_df["group"] = (results_df["Xdays_label"] != results_df["Xdays_label"].shift()).cumsum()
    group_sizes = results_df.groupby("group")["Xdays_label"].transform("size")

    results_df["Xdays_label"] = results_df.apply(
        lambda row: row["Xdays_label"] if group_sizes[row.name] >= trend_size else 0,
        axis=1,
    )
    results_df = results_df.drop(columns=["group"])

    # ======= II. R2 Rank Label =======
    # ------- 1. Apply the labelling from regR2rank -------
    results_df["reg_label"] = regR2rank_labeller(price_series=price_series, params=params)

    # ------- 2. Eliminate the trends that are too small -------
    results_df["group"] = (results_df["reg_label"] != results_df["reg_label"].shift()).cumsum()
    group_sizes = results_df.groupby("group")["reg_label"].transform("size")

    results_df["reg_label"] = results_df.apply(
        lambda row: row["reg_label"] if group_sizes[row.name] >= trend_size else 0,
        axis=1,
    )
    results_df = results_df.drop(columns=["group"])

    # ======= III. Labels combination =======
    # ------- 1. Combine the labels  -------
    results_df["combination_label"] = results_df["Xdays_label"] * 2 + results_df["reg_label"]
    results_df["combination_label"] = results_df["combination_label"].replace(1, np.nan).replace(-1, np.nan)
    results_df["combination_label"] = results_df["combination_label"].fillna(method="ffill")
    results_df["combination_label"] = results_df["combination_label"].replace(2, 1).replace(-2, -1).replace(3, 1).replace(-3, -1)

    # ------- 2. Manage the case of direct change in trend in reg_label -------
    results_df["combination_label"] = results_df.apply(
        lambda row: 0 if row["combination_label"] == 1 and row["reg_label"] == -1 else (0 if row["combination_label"] == -1 and row["reg_label"] == 1 else row["combination_label"]),
        axis=1,
    )

    # ------- 3. Eliminate the trends that are too small -------
    results_df["group"] = (results_df["combination_label"] != results_df["combination_label"].shift()).cumsum()
    group_sizes = results_df.groupby("group")["combination_label"].transform("size")

    results_df["combination_label"] = results_df.apply(
        lambda row: row["combination_label"] if group_sizes[row.name] >= trend_size else 0,
        axis=1,
    )
    results_df = results_df.drop(columns=["group"])

    # ------- 4. Eliminate the last point of each trend -------
    results_df["next_combination_label"] = results_df["combination_label"].shift(-1)
    results_df["combination_label"] = results_df.apply(
        lambda row: row["combination_label"] if row["next_combination_label"] != 0 else 0,
        axis=1,
    )

    label_series = results_df["combination_label"]

    return label_series

