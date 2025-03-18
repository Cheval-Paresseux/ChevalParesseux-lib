import pandas as pd
import numpy as np


#! ==================================================================================== #
#! ============================= BINARY LABELLERS ===================================== #
def right_wrong(preds_labels_df: pd.DataFrame):
    """
    Perform binary labelling of the predictions and labels :
        - 1: Correct predictions (1,1), (-1,-1), (0,0)
        - 0: Misclassifications (1,-1), (-1,1), (1,0), (-1,0), (0,1), (0,-1)
    """
    meta_df = preds_labels_df.copy()

    meta_df["meta_label"] = np.where(meta_df["predictions"] == meta_df["label"], 1, 0)
    label_series = meta_df["meta_label"]

    return label_series

#*____________________________________________________________________________________ #
def trade_lock(preds_labels_df: pd.DataFrame):
    """
    Perform binary labelling of the predictions and labels :
        - 1: Acceptable predictions (1,1), (-1,-1), (0,0), (0,1), (0,-1)
        - 0: Non acceptable predictions (1,-1), (-1,1), (1,0), (-1,0)
    """
    meta_df = preds_labels_df.copy()

    meta_df["meta_label"] = np.where(meta_df["predictions"] == 0, 1, np.where(meta_df["predictions"] == meta_df["label"], 1, 0))
    label_series = meta_df["meta_label"]

    return label_series

#*____________________________________________________________________________________ #
def right_wrong_noZero(preds_labels_df: pd.DataFrame):
    """
    Perform binary labelling of the predictions and labels :
        - 1: Correct predictions (1,1), (-1,-1)
        - 0: Misclassifications (0,0), (1,-1), (-1,1), (1,0), (-1,0), (0,1), (0,-1)
    """
    meta_df = preds_labels_df.copy()

    meta_df["meta_label"] = np.where((meta_df["predictions"] == meta_df["label"]) & (meta_df["predictions"] != 0), 1, 0)
    label_series = meta_df["meta_label"]

    return label_series



#! ==================================================================================== #
#! ============================= TRINARY LABELLERS ==================================== #
def good_bad_ugly(preds_labels_df: pd.DataFrame):
    """
    Perform trinary labelling of the predictions and labels :
        - 1: Good predictions (1,1), (-1,-1), (0,0)
        - 0: Bad predictions (1,0), (-1,0), (0,1), (0,-1)
        - -1: Ugly predictions (1,-1), (-1,1)
    """
    meta_df = preds_labels_df.copy()

    meta_df["meta_label"] = np.where(
        meta_df["predictions"] == meta_df["label"],
        1,
        np.where(((meta_df["predictions"] == -1) & (meta_df["label"] == 1)) | ((meta_df["predictions"] == 1) & (meta_df["label"] == -1)), -1, 0),
    )
    label_series = meta_df["meta_label"]

    return label_series

#*____________________________________________________________________________________ #
def gbu_extended(preds_labels_df: pd.DataFrame):
    """
    Perform trinary labelling of the predictions and labels :
        - 1: Good predictions (1,1), (-1,-1), (0,0)
        - 0: Bad predictions (0,1), (0,-1)
        - -1: Ugly predictions (1,-1), (-1,1), (1,0), (-1,0)
    """
    meta_df = preds_labels_df.copy()

    meta_df["meta_label"] = np.where(
        meta_df["predictions"] == meta_df["label"],
        1,
        np.where((meta_df["predictions"] == 0) & (meta_df["label"] != 0), 0, -1),
    )
    label_series = meta_df["meta_label"]

    return label_series

#*____________________________________________________________________________________ #
def good_bad_ugly_noZero(preds_labels_df: pd.DataFrame):
    """
    Perform trinary labelling of the predictions and labels :
        - 1: Good predictions (1,1), (-1,-1)
        - 0: Bad predictions (0,0), (1,0), (-1,0), (0,1), (0,-1)
        - -1: Ugly predictions (1,-1), (-1,1)
    """
    meta_df = preds_labels_df.copy()

    meta_df["meta_label"] = np.where(
        (meta_df["predictions"] == meta_df["label"]) & (meta_df["predictions"] != 0),
        1,
        np.where(((meta_df["predictions"] == -1) & (meta_df["label"] == 1)) | ((meta_df["predictions"] == 1) & (meta_df["label"] == -1)), -1, 0),
    )
    label_series = meta_df["meta_label"]

    return label_series

#*____________________________________________________________________________________ #
def gbu_extended_noZero(preds_labels_df: pd.DataFrame):
    """
    Perform trinary labelling of the predictions and labels :
        - 1: Good predictions (1,1), (-1,-1)
        - 0: Bad predictions (0,0), (0,1), (0,-1)
        - -1: Ugly predictions (1,-1), (-1,1), (1,0), (-1,0)
    """
    meta_df = preds_labels_df.copy()

    meta_df["meta_label"] = np.where(
        (meta_df["predictions"] == meta_df["label"]) & (meta_df["predictions"] != 0),
        1,
        np.where((meta_df["predictions"] == 0) & (meta_df["label"] != 0), 0, -1),
    )
    label_series = meta_df["meta_label"]

    return label_series
