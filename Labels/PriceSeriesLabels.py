import numpy as np
import pandas as pd
from math import gamma, sqrt, pi

from scipy.optimize import fsolve
from scipy.stats import beta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ========================== COMBINATION LABELLER ========================== #
def combination_labeller(price_series: pd.Series, params: dict):
    """
    labelling_params = {
        "size_window_smooth": 10,
        "lambda_smooth": 0.2,
        "trend_size": 10,
        "volatility_threshold": 1.5,
        "horizon": 10,
        "horizon_extension": 1.5,
        "upper_r2_threshold": 0.8,
        "lower_r2_threshold": 0.5,
        "r": 0,
    }
    """
    # ======= 0. Params extraction & Initialization =======
    price_series = price_series.dropna()

    size_window_smooth = params["size_window_smooth"]
    lambda_smooth = params["lambda_smooth"]
    trend_size = params["trend_size"]
    volatility_threshold = params["volatility_threshold"]

    results_df = price_series.to_frame()
    results_df["smooth_close"] = trunc_expon_smooth(price_series, size_window_smooth, lambda_smooth)

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


# ========================== REGR2RANK LABELLER ========================== #
def regR2rank_labeller(price_series: pd.Series, params: dict):
    # ======= I. Extract the parameters =======
    size_window_smooth = int(params["size_window_smooth"])
    lambda_smooth = params["lambda_smooth"]
    horizon = int(params["horizon"])
    horizon_extension = params["horizon_extension"]
    upper_r2_threshold = params["upper_r2_threshold"]
    lower_r2_threshold = params["lower_r2_threshold"]
    r = int(params["r"])

    horizon_max = round(horizon * (1 + horizon_extension))

    # ======= II. Extract the data to numpy array =======
    data_array = np.array(price_series)
    data_close = data_array.copy()

    size_data_array = data_array.shape[0]
    zeros = np.repeat(np.nan, size_data_array)

    # ======= III. Smooth the data using truncated EMA filter =======
    if size_window_smooth > 1:
        ema_close = trunc_expon_smooth(values=data_close, window_size=size_window_smooth, ind_lambda=lambda_smooth)

        nan_count_start = np.sum(np.isnan(ema_close))  # Number of NaNs at the beginning, due the smoothing using rolling window
        nan_count_end = 0
    else:
        # As the window size is 1, the smoothed data is the same as the original data
        ema_close = data_close

        nan_count_start = 0
        nan_count_end = 0

    # ======= IV. Labelling =======
    # ------- 1. Dataframe to store the results -------
    results_df_cols = ["position", "ema_close", "horizon", "slope", "r2", "label"]
    results_df = np.c_[zeros, ema_close, zeros, zeros, zeros, zeros]

    # ------- 2. Start the labelling process -------
    starting_index = 1 + nan_count_start
    iterating_index = starting_index
    while iterating_index <= size_data_array - horizon - nan_count_end + 1:
        best_r2 = -1

        # ------- i. Test the regression on different horizon size -------
        horizon_max_test = min(horizon_max, size_data_array - nan_count_end - iterating_index + 1)
        for horizon_test in range(horizon, horizon_max_test + 1):
            index_test = np.array(range(0, horizon_test)) + iterating_index

            y_close = ema_close[index_test - 1]

            X = index_test.reshape(-1, 1)
            Y = y_close

            regression_model = LinearRegression().fit(X, Y)
            y_pred = regression_model.predict(X)
            slope = regression_model.coef_[0]
            r2 = r2_score(Y, y_pred)

            # Keep the best configuration only
            if r2 > best_r2:
                best_r2 = r2
                best_horizon = horizon_test
                best_slope = slope

        results_df[iterating_index - 1, results_df_cols.index("position")] = iterating_index
        results_df[iterating_index - 1, results_df_cols.index("horizon")] = best_horizon
        results_df[iterating_index - 1, results_df_cols.index("slope")] = best_slope
        results_df[iterating_index - 1, results_df_cols.index("r2")] = best_r2
        results_df[iterating_index - 1, results_df_cols.index("label")] = np.sign(best_slope)

        iterating_index += 1

    # ======= V. Final step: labelling the points in descending order of R2 =======
    label_reg = np.repeat(0, size_data_array)
    index_ordered = np.array(pd.Series(results_df[:, results_df_cols.index("r2")]).sort_values(ascending=False).index)
    results_df_ordered = results_df[index_ordered, :]
    results_df_cols_ordered = results_df_cols.copy()

    for i in range(1, size_data_array + 1):
        position = results_df_ordered[i - 1, results_df_cols_ordered.index("position")]
        horizon = results_df_ordered[i - 1, results_df_cols_ordered.index("horizon")]
        slope = results_df_ordered[i - 1, results_df_cols_ordered.index("slope")]
        r2 = results_df_ordered[i - 1, results_df_cols_ordered.index("r2")]

        signal = np.sign(slope)

        if not np.isnan(r2):
            keep_label = False
            if signal == 1 and results_df_ordered[i - 1, results_df_cols_ordered.index("r2")] >= upper_r2_threshold:
                keep_label = True

            if signal == -1 and results_df_ordered[i - 1, results_df_cols_ordered.index("r2")] >= lower_r2_threshold:
                keep_label = True

            if keep_label:
                temporary_index = np.array(range(int(horizon))) + position.astype(int) - 1
                non_zeros_index = np.argwhere(label_reg[temporary_index] != 0).reshape(-1)

                if len(non_zeros_index) == 0:
                    label_reg[temporary_index] = signal
                else:
                    if np.sum(label_reg[temporary_index[non_zeros_index]] != signal) == 0:
                        label_reg[temporary_index] = signal

    results_df = np.c_[results_df, label_reg.astype(int)]
    results_df_cols.append("label_reg")

    # ======= VI. Aggregation of labels =======
    results_df = np.c_[results_df, labels_aggregator(label=label_reg.astype(int), r=r).astype(int)]
    results_df_cols.append("labelp")

    final_results_df = pd.DataFrame(results_df, columns=results_df_cols).label_reg.values

    return final_results_df


# ========================== TRIPLE BARRIER LABELLER ========================== #
def tripleBarrier_labeller(price_series: pd.Series, params: dict):
    # ======= 0. Auxiliary functions =======
    def observed_volatility(price_series: pd.Series, window: int):
        """
        Computes rolling window volatility using percentage returns.

        Args:
            price_series (pd.Series): Price series of the asset
            window (int): Window for the rolling computation

        Returns:
            volatility_series (pd.Series): Rolling volatility series
        """
        returns_series = price_series.pct_change().fillna(0)
        volatility_series = returns_series.rolling(window).std() * np.sqrt(window)

        return volatility_series

    # ======= I. Compute volatility target =======
    upper_barrier = params["upper_barrier"]
    lower_barrier = params["lower_barrier"]
    vertical_barrier = params["vertical_barrier"]
    volatility_function = params["volatility_function"]

    if volatility_function == "observed":
        volatility_series = observed_volatility(price_series=price_series, window=vertical_barrier)

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

        # III.4 Check if the horiazontal barriers have been hit
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

        # III.5 Label the events base on the first event that occurs
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


# ========================== LOOK FORWARD LABELLER ========================== #
def lookForward_labeller(price_series: pd.Series, params: dict):
    # ======= 0. Params extraction & Initialization =======
    price_series = price_series.dropna()

    size_window_smooth = params["size_window_smooth"]
    lambda_smooth = params["lambda_smooth"]
    trend_size = params["trend_size"]
    volatility_threshold = params["volatility_threshold"]

    results_df = price_series.to_frame()
    results_df["smooth_close"] = trunc_expon_smooth(price_series, size_window_smooth, lambda_smooth)

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

    label_series = results_df["Xdays_label"]

    return label_series


# ========================================================================= #
# ========================== AUXILIARY FUNCTIONS ========================== #
def labels_aggregator(label: np.array, r: int):
    """
    This function aggregates the labels based on the conviction metric.

    Args:
        label (np.array): The array containing the labels.
        r (int): The parameter between 0 and 1 for the calculation of the conviction function.

    Returns:
        filtered_labels (np.array): The array containing the filtered labels.
    """
    if r == 0:
        filtered_labels = label
    else:
        # ======= I. Identify the segments of distinct labels along the series =======
        segments_df, segments_df_col = get_label_segments(label=label, r=r)

        # ======= II. Aggregate the labels based on the conviction metric =======
        filtered_segments_df, df_segmentos_filt_col = funde_segmentos_rotulos(r, segments_df, segments_df_col)

        # ======= III. Create the final labels =======
        filtered_labels = np.repeat(np.nan, len(label))
        for i in range(filtered_segments_df.shape[0]):
            initial = filtered_segments_df[i, df_segmentos_filt_col.index("idstart")].astype(int)
            final = filtered_segments_df[i, df_segmentos_filt_col.index("idend")].astype(int) + 1
            filtered_labels[initial:final] = filtered_segments_df[i, df_segmentos_filt_col.index("label")]

    return filtered_labels


# --------------------------------------------------------------------------------------------------------------
def get_label_segments(label: np.array, r: int):
    """
    This function identifies the segments of distinct labels along the series.

    Args:
        label (np.array): The array containing the labels.
        r (int): The parameter between 0 and 1 for the calculation of the conviction function.

    Returns:
        segments_df (np.array): The array containing the segments.
        segments_df_col (list): The list containing the columns of the segments dataframe
    """
    # ======= I. Initialize the intermediate variables =======
    non_nan_index = np.argwhere(~np.isnan(label)).reshape(-1)
    values_quantity = len(non_nan_index)

    temporary_label = np.copy(label)
    unique_labels = np.unique(temporary_label[~np.isnan(temporary_label)]).astype("int")
    unique_labels_names = [f"count{i}" for i in unique_labels]

    # ======= II. Initialize the Segment dataframe =======
    segments_df = np.array([])
    segments_df_col = []
    first_segment_check = True
    previous_label = temporary_label[non_nan_index[0]]
    segment_start_index = non_nan_index[0]

    # ======= III. Iterate over the valid indices labels to identify the segments =======
    # A segment ends when the current label differs from the previous label or when it reaches the last valid index.
    for i in range(non_nan_index[1] + 1, non_nan_index[values_quantity - 1] + 2):
        if i == non_nan_index[values_quantity - 1] + 1 or temporary_label[i] != previous_label:
            # ------- i. Calculate the frequencies of each label in the segment -------
            segment_end_index = i - 1
            segment_size = segment_end_index - segment_start_index + 1

            (_, observed_frequencies) = get_frequency(
                data_array=temporary_label[get_range(a=segment_start_index, b=segment_end_index)],
                possible_values_array=unique_labels,
            )

            # ------- ii. Calculate the conviction and loss for the segment -------
            error = 0
            segment_conviction = conviction_metrics(
                nb_elements=segment_size,
                nb_misclassified=error,
                convexity_factor=r,
                possible_outcomes=2,
            )
            segment_loss = segment_conviction * segment_size

            # ------- iii. Initialize the intermediate segment dataframe -------
            intermediate_df = np.array(
                [
                    [
                        segment_start_index,
                        segment_end_index,
                        segment_size,
                        previous_label,
                        error,
                        segment_conviction,
                        segment_loss,
                    ]
                ]
            )
            names_intermediate_df = [
                "start_index",
                "end_index",
                "size",
                "label",
                "error",
                "conviction",
                "loss",
            ]

            # ------- iv. Add the frequencies to the intermediate segment dataframe -------
            columns_intermediate_df = names_intermediate_df + unique_labels_names
            intermediate_df = np.c_[intermediate_df, observed_frequencies.reshape(1, -1)]

            # ------- v. Update the segment dataframe -------
            if first_segment_check:
                segments_df = intermediate_df
                segments_df_col = columns_intermediate_df
                first_segment_check = False
            else:
                segments_df = np.r_[segments_df, intermediate_df]

            # ------- vi. Update the segment variables for the next iteration -------
            if i < non_nan_index[values_quantity - 1] + 1:
                previous_label = temporary_label[i]
                segment_start_index = i

    return segments_df, segments_df_col


# --------------------------------------------------------------------------------------------------------------
def funde_segmentos_rotulos(r: int, segments_df_input: np.array, segments_df_col_input: list):
    # ======= I. Initialize the intermediate variables =======
    segments_df = np.copy(segments_df_input)
    segments_df_col = segments_df_col_input.copy()

    NewStat = np.copy(segments_df)
    NewStat_col = segments_df_col.copy()
    unique_labels = np.unique(segments_df[:, segments_df_col.index("label")]).astype(int)

    unique_labels_names = [i for i in segments_df_col if "count" in i]
    unique_labels_quantity = len(unique_labels_names)

    # ======= II. Initialize the final variables =======
    #  The loop continues until there are no more intervals joins with positive gain
    runs_count = 0
    while True:
        # ------- i. Initialize the intermediate variables -------
        runs_count = runs_count + 1
        quantity_label = NewStat.shape[0]

        if quantity_label == 1:
            break

        max_distance = 5
        nmax = (quantity_label - 1) * max_distance

        # StatMerge: data frame auxiliar para computar os ganhos em
        # agrupar segmentos de mesma classe

        StatMerge_col = [
            "stint1",
            "stint2",
            "stclass",
            "stvini",
            "stvfim",
            "stsize",
            "sterror",
            "stconviction",
            "stloss",
            "stgain",
        ]
        StatMerge = np.zeros((nmax, 9))
        StatMerge = np.c_[StatMerge, np.repeat(-1e10, nmax)]

        stcounts = np.zeros((nmax, unique_labels_quantity))

        StatMerge = np.c_[StatMerge, stcounts]
        StatMerge_col = StatMerge_col + unique_labels_names

        qtpart = 0

        for i1 in range(1, (quantity_label - max_distance) + 1):  # i1 = 1            i1 += 1
            # trabalhamos com segmentos em tamanho decrescente
            for i2 in get_range(min(quantity_label, i1 + max_distance), i1 + 1):  # i2 = 6            i2 -= 1
                # print(f"runs = {runs}  --  i1 = {i1}  --  i2 = {i2}  --   {NewStat.shape[0]}")

                # Somente agrupamentos de intervalos cujos extremos sejam de mesma
                # classe serao considerados
                if NewStat[i1 - 1, NewStat_col.index("label")] == NewStat[i2 - 1, NewStat_col.index("label")]:
                    qtpart = qtpart + 1
                    StatMerge[qtpart - 1, StatMerge_col.index("stint1")] = i1 - 1
                    StatMerge[qtpart - 1, StatMerge_col.index("stint2")] = i2 - 1
                    StatMerge[qtpart - 1, StatMerge_col.index("stclass")] = NewStat[i1 - 1, NewStat_col.index("label")]
                    StatMerge[qtpart - 1, StatMerge_col.index("stvini")] = NewStat[i1 - 1, NewStat_col.index("idstart")]
                    StatMerge[qtpart - 1, StatMerge_col.index("stvfim")] = NewStat[i2 - 1, NewStat_col.index("idend")]

                    # print(StatMerge.head(6).to_string())

                    idxrange = np.array(range(i1, i2 + 1))
                    SizesInterv = NewStat[idxrange - 1, NewStat_col.index("size")]
                    LossInterv = NewStat[idxrange - 1, NewStat_col.index("loss")]

                    ContInterv = NewStat[
                        i1 - 1 : i2,
                        np.array([NewStat_col.index(i) for i in unique_labels_names]),
                    ]

                    # print(NewStat.head(5).to_string())
                    countclass = np.sum(ContInterv, axis=0)

                    # Classe do Segmento será classe dos extremos
                    # idclass = classes[classes == StatMerge[qtpart-1, StatMerge_col.index("stclass")]][0]
                    idclass = np.argwhere(unique_labels == StatMerge[qtpart - 1, StatMerge_col.index("stclass")])[0][0]

                    # Erro de Classificacao
                    erro = np.sum(countclass) - countclass[idclass]

                    StatMerge[qtpart - 1, StatMerge_col.index("stsize")] = np.sum(SizesInterv)
                    StatMerge[qtpart - 1, StatMerge_col.index("sterror")] = erro
                    StatMerge[qtpart - 1, StatMerge_col.index("stconviction")] = conviction_metrics(
                        StatMerge[qtpart - 1, StatMerge_col.index("stsize")],
                        StatMerge[qtpart - 1, StatMerge_col.index("sterror")],
                        r,
                        qtclass=2,
                    )
                    StatMerge[qtpart - 1, StatMerge_col.index("stloss")] = StatMerge[qtpart - 1, StatMerge_col.index("stconviction")] * StatMerge[qtpart - 1, StatMerge_col.index("stsize")]
                    StatMerge[qtpart - 1, StatMerge_col.index("stgain")] = np.sum(LossInterv) - StatMerge[qtpart - 1, StatMerge_col.index("stloss")]

                    StatMerge[
                        qtpart - 1,
                        [StatMerge_col.index(i) for i in unique_labels_names],
                    ] = countclass
                # End for i2
            # End for i1

        # --------------------------------------------------
        # Mantem apenas agrupamentos com ganhos negativos, se nao houver tais agrupamentos,  encerrar
        temp = np.argwhere(StatMerge[:, StatMerge_col.index("stgain")] > 0).reshape(-1)
        StatMerge = StatMerge[temp, :]

        if len(StatMerge) == 0:
            break

        qtpart = StatMerge.shape[0]

        # Este trecho é para evitar "conflitos" entre segmentos, ou seja,
        # que ocorram dois "merges" sobre os mesmos pontos da serie
        i1 = 1

        while i1 < qtpart:
            # Soh nos preocupam os segmentos com ganho positivo,
            # pois aqueles com ganho negativo nao serao agrupados

            if StatMerge[i1 - 1, StatMerge_col.index("stgain")] > 0:
                i2 = i1 + 1
                while True:
                    if StatMerge[i2 - 1, StatMerge_col.index("stint1")] > StatMerge[i1 - 1, StatMerge_col.index("stint2")]:
                        break
                    i2 = i2 + 1
                    if i2 > qtpart:
                        break
                # volta uma posicao (pois o ultimo segmento visitado nao tinha sobreposicao)
                i2 = i2 - 1

                if i2 > i1:
                    # Houve segmento sobrepostos?
                    # posicao relativa do melhor agrupamento dentro do intervalo i1:i2
                    idxbest = np.argmax(StatMerge[i1 - 1 : i2, StatMerge_col.index("stgain")])
                    # Os segmentos no intervalo i1:i2 que nao corresponderem ao otimo
                    # terao seus ganhos transformados em negativos
                    tem = np.array(range(i1 - 1, i2))
                    tem = np.delete(tem, idxbest)

                    StatMerge[tem, StatMerge_col.index("stgain")] = -1e10
                    i1 = i2
                else:
                    i1 = i1 + 1
            else:
                i1 = i1 + 1
            # Fim if StatMerge["stgain"][i1-1] > 0
        # Fim while i1 < qtpart:

        # ================================================================================
        # Verifica se há segmentos que possam ser agrupados
        idxpos = np.argwhere(StatMerge[:, StatMerge_col.index("stgain")] > 0).reshape(-1)

        if len(idxpos) == 0:
            break

        for idp in idxpos:  # idp = idxpos[0]
            idxint1 = StatMerge[idp, StatMerge_col.index("stint1")].astype(int)
            idxint2 = StatMerge[idp, StatMerge_col.index("stint2")].astype(int)

            NewStat[idxint1, NewStat_col.index("idstart")] = StatMerge[idp, StatMerge_col.index("stvini")]
            NewStat[idxint1, NewStat_col.index("idend")] = StatMerge[idp, StatMerge_col.index("stvfim")]
            NewStat[idxint1, NewStat_col.index("size")] = StatMerge[idp, StatMerge_col.index("stsize")]
            NewStat[idxint1, NewStat_col.index("error")] = StatMerge[idp, StatMerge_col.index("sterror")]
            NewStat[idxint1, NewStat_col.index("convic")] = StatMerge[idp, StatMerge_col.index("stconviction")]
            NewStat[idxint1, NewStat_col.index("loss")] = StatMerge[idp, StatMerge_col.index("stloss")]

            new_ind_temp = [NewStat_col.index(f"{k}") for k in unique_labels_names]
            old_ind_temp = [StatMerge_col.index(f"{k}") for k in unique_labels_names]

            NewStat[idxint1, new_ind_temp] = StatMerge[idp, old_ind_temp]

            # Coloca um sentinela na coluna idstart para posterior remocao dos registros
            NewStat[idxint1 + 1 : idxint2 + 1, NewStat_col.index("idstart")] = -1

        # NewStat = NewStat[NewStat["idstart"] >= 0]
        var_temp = np.argwhere(NewStat[:, NewStat_col.index("idstart")] >= 0).reshape(-1)
        NewStat = NewStat[var_temp, :]

        # NewStat_ids = np.argwhere(NewStat[:, NewStat_col.index("idstart")].reshape(-1) >= 0).reshape(-1)
        # NewStat = np.argwhere(NewStat[:, NewStat_col.index("idstart")].reshape(-1) >= 0).reshape(-1)
    # Fim While True

    return NewStat, NewStat_col


# --------------------------------------------------------------------------------------------------------------
def WMA(values: np.array, weight_range: np.array):
    """
    Perform a weighted moving average on a numpy array.

    Args:
        values (np.array): The array of values to be averaged.
        weights (int or np.array): The weights to be used in the average. If an integer is passed, the function will use the last n values to calculate the average.

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


# --------------------------------------------------------------------------------------------------------------
def trunc_expon_smooth(values, window_size, ind_lambda):
    """
    Perform a weighted moving average on a numpy array using a truncated exponential function. The objective is to give more importance to the latest values in the array.

    Args:
        values (np.array): The array of values to be averaged.
        window_size (int): The size of the window to be used in the moving average.
        ind_lambda (float): The lambda parameter for the exponential function.

    Returns:
        wma (np.array): The array of weighted averages.
    """
    # ======= I. Create the weights using a truncated exponential function =======
    weight_range = [(1 - ind_lambda) ** (i - 1) for i in range(1, window_size + 1)]
    weight_range.reverse()
    weight_range = np.array(weight_range)

    # ======= II. Perform the weighted moving average =======
    wma = WMA(values, weight_range)

    return wma


# --------------------------------------------------------------------------------------------------------------
def t_studen(x, v):
    a = gamma((v + 1) / 2) / ((sqrt(v * pi)) * gamma(v / 2))
    b = (1 + (x**2) / v) ** (-(v + 1) / 2)
    return a * b


# --------------------------------------------------------------------------------------------------------------
def dt(x, v):
    return t_studen(x, v)


# --------------------------------------------------------------------------------------------------------------
def conviction_metrics(nb_elements, nb_misclassified, convexity_factor, possible_outcomes):
    """
    This function calculates the conviction value for a given interval.
    It helps determine the likelihood or reliability of the segment being accurately labeled or classified based on the beta distribution.

    Args:
        nb_elements (int): The number of elements in the interval.
        nb_error (int): The number of misclassified elements in the interval.
        convexity_factor (float): The convexity factor.
        quantity_possible_outcomes (int): The number of possible outcomes.

    Returns:
        float: The conviction value in the (0, 1) interval.
    """

    # ======= I. Define the function to solve =======
    def f(x):
        return (
            1
            - x**convexity_factor
            - beta.cdf(
                x,
                nb_misclassified + possible_outcomes - 1,
                nb_elements - nb_misclassified + 1,
                loc=0,
                scale=1,
            )
        )

    # ======= II. Solve the equation =======
    result = fsolve(f, 0.5)[0]

    return result


# --------------------------------------------------------------------------------------------------------------
def get_range(a, b):
    """
    This function returns a range from a to b, including both a and b, while considering the direction of the range.

    Args:
        a (int): The starting point of the range.
        b (int): The ending point of the range.

    Returns:
        range: The range from a to b.
    """
    if a <= b:
        # The range is increasing
        result = range(a, b + 1)
    else:
        # The range is decreasing
        result = range(a, b - 1, -1)

    return result


# --------------------------------------------------------------------------------------------------------------
def get_frequency(data_array, possible_values_array):
    """
    This function calculates the frequency of each possible outcome in the data array.

    Args:
        data_array (np.array): The array containing the data.
        possible_outcomes_array (np.array): The array containing the possible outcomes.

    Returns:
        unique_values (np.array): The array containing the unique values.
        frequencies (np.array): The array containing the frequencies of each unique value.
    """
    # ======= I. Identify the unique values and their frequency in the data =======
    possible_values = np.array(possible_values_array).astype("int")
    (unique_values, frequencies) = np.unique(data_array, return_counts=True)

    # ======= II. Adjustments for values non-present in the data =======
    for i in possible_values:
        if i not in unique_values:
            unique_values = np.append(unique_values, i)
            frequencies = np.append(frequencies, 0)

    # ======= III. Sort the values and frequencies =======
    pos = np.array([np.where(unique_values == i) for i in possible_values]).reshape(-1)
    unique_values = unique_values[pos]
    frequencies = frequencies[pos]

    return unique_values, frequencies


# --------------------------------------------------------------------------------------------------------------
def update(data, atual):
    data = data.copy()
    for ind in atual.index:
        for col in atual.columns:
            i = list(data.index).index(ind)
            j = list(data.columns).index(col)
            data.iloc[i, j] = atual[col][ind]
    return data
