import os
import sys

sys.path.append(os.path.abspath("../"))
import Labelling.TS_Labels as ts_labels
import Labelling.META_Labels as meta_labels

from tqdm import tqdm


def ts_labelling(individual_dfs_list: list, labelling_params: dict, labelling_method: str):
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
        "upper_barrier": 1,
        "lower_barrier": 1,
        "vertical_barrier": 5,
        "volatility_function": "observed"
    }
    """
    # ======= I. Initialize input and output =======
    individual_dfs = individual_dfs_list.copy()
    labeled_dfs_list = []

    # ======= II. Apply the labelling method =======
    for asset_df in tqdm(individual_dfs):
        # II.1. Initialize the dataframe with the asset price
        asset_name = asset_df.columns[0]
        if len(asset_name) > 10:
            raise ValueError(f"The column 0 of the dataframe should be the asset price series, not {asset_name}. Please put the price series in the first column with a name < 10 characters.")

        labeled_df = asset_df.copy()
        labeled_df = labeled_df.astype(float)
        labeled_df = labeled_df.dropna(axis=0)

        # II.2. Apply the chosen labelling method
        if labelling_method == "combination":
            labeled_series = ts_labels.combination_labeller(price_series=asset_df[asset_name], params=labelling_params)
            labeled_df["label"] = labeled_series

        elif labelling_method == "regR2rank":
            labeled_series = ts_labels.regR2rank_labeller(price_series=asset_df[asset_name], params=labelling_params)
            labeled_df["label"] = labeled_series

        elif labelling_method == "lookForward":
            labeled_series = ts_labels.lookForward_labeller(price_series=asset_df[asset_name], params=labelling_params)
            labeled_df["label"] = labeled_series

        elif labelling_method == "tripleBarrier":
            labeled_series = ts_labels.tripleBarrier_labeller(price_series=asset_df[asset_name], params=labelling_params)
            labeled_df["label"] = labeled_series

        else:
            raise ValueError("Invalid labelling method")

        # II.3. Save the labeled dataframe
        labeled_dfs_list.append(labeled_df)

    return labeled_dfs_list


# -----------------------------------------------------------
def meta_labelling(individual_dfs_list: list, labelling_method: str):
    # ======= I. Initialize input and output =======
    individual_dfs = individual_dfs_list.copy()
    labeled_dfs_list = []

    # ======= II. Apply the labelling method =======
    for asset_df in individual_dfs:
        # II.1. Initialize the dataframe with the asset price
        asset_name = asset_df.columns[0]
        if len(asset_name) > 10:
            raise ValueError(f"The column 0 of the dataframe should be the asset price series, not {asset_name}. Please put the price series in the first column with a name < 10 characters.")
        if "predictions" not in asset_df.columns:
            raise ValueError("The dataframe should contain the predictions column.")

        labeled_df = asset_df.copy()
        labeled_df = labeled_df.astype(float)
        labeled_df = labeled_df.dropna(axis=0)

        # II.2. Apply the chosen labelling method
        if labelling_method == "right_wrong":
            label_series = meta_labels.right_wrong(asset_df)
            labeled_df["meta_label"] = label_series
        elif labelling_method == "right_wrong_noZero":
            label_series = meta_labels.right_wrong_noZero(asset_df)
            labeled_df["meta_label"] = label_series
        elif labelling_method == "trade_lock":
            label_series = meta_labels.trade_lock(asset_df)
            labeled_df["meta_label"] = label_series

        # ------- Trinary Meta Labelling -------
        elif labelling_method == "good_bad_ugly":
            label_series = meta_labels.good_bad_ugly(asset_df)
            labeled_df["meta_label"] = label_series
        elif labelling_method == "good_bad_ugly_noZero":
            label_series = meta_labels.good_bad_ugly_noZero(asset_df)
            labeled_df["meta_label"] = label_series
        elif labelling_method == "gbu_extended":
            label_series = meta_labels.gbu_extended(asset_df)
            labeled_df["meta_label"] = label_series
        elif labelling_method == "gbu_extended_noZero":
            label_series = meta_labels.gbu_extended_noZero(asset_df)
            labeled_df["meta_label"] = label_series

        # II.3. Save the labeled dataframe
        labeled_dfs_list.append(labeled_df)

    return labeled_dfs_list
