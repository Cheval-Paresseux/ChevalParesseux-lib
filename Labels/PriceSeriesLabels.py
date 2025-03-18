import sys
sys.path.append("../")
from Models import LinearRegression as reg
from Labels import auxiliary as aux

import numpy as np
import pandas as pd


#! ==================================================================================== #
#! =================================== LABELLERS ====================================== #
def tripleBarrier_labeller(price_series: pd.Series, params: dict):
    # ======= 0. Params extraction =======
    upper_barrier = params["upper_barrier"]
    lower_barrier = params["lower_barrier"]
    vertical_barrier = params["vertical_barrier"]

    # ======= I. Compute volatility target =======
    p_series = price_series.dropna().copy()
    volatility_series = aux.get_volatility(price_series=p_series, window=vertical_barrier)

    # ======= II. Initialize the labeled series and trade side =======
    labels_series = pd.Series(index=p_series.index, dtype=int)
    trade_side = 0

    # ======= III. Iterate through the price series =======
    for index in p_series.index:
        # III.1 Extract the future prices over the horizon
        start_idx = p_series.index.get_loc(index)
        end_idx = min(start_idx + vertical_barrier, len(p_series))
        future_prices = p_series.iloc[start_idx:end_idx]

        # III.2 Compute the range of future returns over the horizon
        max_price = future_prices.max()
        min_price = future_prices.min()

        max_price_index = future_prices.idxmax()
        min_price_index = future_prices.idxmin()

        max_return = (max_price - p_series.loc[index]) / p_series.loc[index]
        min_return = (min_price - p_series.loc[index]) / p_series.loc[index]

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
                labels_series.loc[index] = 1
            else:
                labels_series.loc[index] = -1

        elif long_event and not short_event:  # If only long event occurs
            labels_series.loc[index] = 1

        elif short_event and not long_event:  # If only short event occurs
            labels_series.loc[index] = -1

        else:  # If no event occurs (vertical hit)
            labels_series.loc[index] = 0

        # III.6 Update the trade side
        trade_side = labels_series.loc[index]

    return labels_series

#*____________________________________________________________________________________ #
def lookForward_labeller(price_series: pd.Series, params: dict):
    # ======= 0. Params extraction =======
    size_window_smooth = params["size_window_smooth"]
    lambda_smooth = params["lambda_smooth"]
    trend_size = params["trend_size"]
    volatility_threshold = params["volatility_threshold"]

    # ======= I. Prepare Series =======
    p_series = price_series.dropna().copy()
    ewma_series = aux.exponential_weightedMA(price_series=p_series, window=size_window_smooth, ind_lambda=lambda_smooth)

    # ======= I. Significant look forward Label =======
    # ------- 1. Get the moving X days returns and the moving X days volatility -------
    Xdays_returns = (ewma_series.shift(-size_window_smooth) - ewma_series) / ewma_series
    Xdays_vol = Xdays_returns.rolling(window=size_window_smooth).std()

    # ------- 2. Compare the X days returns to the volatility  -------
    Xdays_score = Xdays_returns / Xdays_vol
    Xdays_label = Xdays_score.apply(lambda x: 1 if x > volatility_threshold else (-1 if x < -volatility_threshold else 0))

    # ------- 3. Eliminate the trends that are too small -------
    labels_series = aux.trend_filter(label_series=Xdays_label, window=trend_size)

    return labels_series

#*____________________________________________________________________________________ #
def regR2rank_labeller(price_series: pd.Series, params: dict):
    # ======= I. Extract the parameters =======
    size_window_smooth = int(params["size_window_smooth"])
    lambda_smooth = params["lambda_smooth"]
    horizon = int(params["horizon"])
    horizon_extension = params["horizon_extension"]
    r2_threshold = params["r2_threshold"]
    trend_size = int(params["trend_size"])

    # ======= II. Initialize the series =======
    p_series = price_series.dropna().copy()
    ewma_series = aux.exponential_weightedMA(price_series=p_series, window=size_window_smooth, ind_lambda=lambda_smooth)
    nb_elements = len(ewma_series)

    labels_series = pd.Series(0, index=p_series.index, dtype=int)  # Initialise à 0

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
                labels_series.iloc[idx] = 1 if slope > 0 else -1  # Correction ici
    
    # ------- 3. Eliminate the trends that are too small -------
    labels_series = aux.trend_filter(label_series=labels_series, window=trend_size)

    return labels_series

#*____________________________________________________________________________________ #
def boostedLF_labeller(price_series: pd.Series, params_lF: dict, params_r2: dict):
    # ======= 0. Params extraction =======
    trend_size = params_lF["trend_size"]
    
    # ======= I. Extract Labels =======
    lookForward_labels = lookForward_labeller(price_series=price_series, params=params_lF)
    regR2rank_labels = regR2rank_labeller(price_series=price_series, params=params_r2)
    
    # ======= II. Linking Trend Holes in regR2rank =======
    regR2rank_labels = regR2rank_labels.replace(0, np.nan)
    forward = regR2rank_labels.ffill()
    backward = regR2rank_labels.bfill()
    regR2rank_labels = forward + backward
    regR2rank_labels = regR2rank_labels.replace(1, 0).replace(-1, 0).replace(2, 1).replace(-2, -1)

    # ======= III. Labels Ensemble =======
    # ------- 1. Combine the labels using lookForward as base -------
    ensemble_labels = lookForward_labels * 2 + regR2rank_labels
    ensemble_labels = ensemble_labels.replace(1, np.nan).replace(-1, np.nan)
    ensemble_labels = ensemble_labels.fillna(method="ffill")
    ensemble_labels = ensemble_labels.replace(2, 1).replace(-2, -1).replace(3, 1).replace(-3, -1)

    # ------- 2. Manage the case of direct change in trend in reg_label -------
    mask_positive_to_negative = (ensemble_labels == 1) & (regR2rank_labels == -1)
    mask_negative_to_positive = (ensemble_labels == -1) & (regR2rank_labels == 1)
    ensemble_labels[mask_positive_to_negative | mask_negative_to_positive] = 0

    # ------- 3. Eliminate the trends that are too small -------
    labels_series = aux.trend_filter(label_series=ensemble_labels, window=trend_size)

    # ------- 4. Eliminate the last point of each trend -------
    next_label = labels_series.shift(-1)
    labels_series[next_label == 0] = 0

    return labels_series

