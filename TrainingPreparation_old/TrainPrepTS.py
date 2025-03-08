import os
import sys

sys.path.append(os.path.abspath("../"))
import MachineLearning.ClusterizersTS as clusterizers

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


def vertical_stacking(training_data_dfs_list: list, rebalancing_strategy: str):
    # ======= I. Initialize input and output =======
    training_data = pd.DataFrame()
    for asset_df in training_data_dfs_list:
        renamed_df = asset_df.copy()
        renamed_df.rename(columns={renamed_df.columns[0]: "asset"}, inplace=True)
        training_data = pd.concat([training_data, renamed_df], axis=0, ignore_index=True)

    if rebalancing_strategy is not None:
        balanced_training_data = rebalance_classes(training_data=training_data, strategy=rebalancing_strategy)
    else:
        balanced_training_data = training_data

    balanced_training_data = balanced_training_data.dropna(axis=0)

    return balanced_training_data


# -----------------------------------------------------------------------------
def rebalance_classes(training_data: pd.DataFrame, strategy: str):
    # ======= I. Initialize the input =======
    X = training_data.drop(columns=["label"]).copy()
    y = training_data["label"].copy()

    # ======= II. Choose Sampler =======
    if strategy == "oversample":
        sampler = RandomOverSampler(random_state=69)

    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=69)

    elif strategy == "smoteenn":
        sampler = SMOTEENN(random_state=69)

    else:
        raise ValueError("Invalid strategy. Choose from 'oversample', 'undersample', or 'smoteenn'.")

    # ======= III. Resample the data =======
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # ======= IV. Create a new DataFrame with the resampled data =======
    resampled_data = pd.DataFrame(X_resampled, columns=X_resampled.columns).assign(label=y_resampled)

    return resampled_data


# -----------------------------------------------------------------------------
def pca_selection(training_data_df: pd.DataFrame, n_components: int):
    # ======= I. Initialize the input =======
    X = training_data_df.drop(columns=["asset", "label"]).copy()
    X = X.dropna(axis=0)

    # ======= II. Perform PCA to sort the features by importance =======
    # III.1 Standardize the data because PCA is sensitive to the scale of the features
    scaler = StandardScaler()
    X_standardized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # III.2 Perform PCA
    pca = PCA(n_components=n_components, random_state=69)
    pca.fit(X_standardized)

    # ======= III. Get the feature importance from the PCA loadings =======
    loadings = np.abs(pca.components_)
    feature_importance = loadings.sum(axis=0)
    importance_df = pd.DataFrame({"feature": X_standardized.columns, "importance": feature_importance})

    importance_df = importance_df.sort_values(by="importance", ascending=False)
    top_features = importance_df.head(n_components)["feature"].tolist()

    # ======= IV. Create a new DataFrame with the selected features =======
    # IV.1 Keep only the selected features
    filtered_training_data = X[top_features].copy()

    # IV.2 Add the asset and label columns and drop NaN values from lagged features
    filtered_training_data["asset"] = training_data_df["asset"]
    filtered_training_data["label"] = training_data_df["label"]

    # IV.3 Drop NaN values
    filtered_training_data = filtered_training_data.dropna(axis=0)

    return filtered_training_data


# -----------------------------------------------------------------------------
def rfImportance_selection(training_data_df: pd.DataFrame, n_components: int):
    # ======= I. Initialize the input =======
    X = training_data_df.drop(columns=["asset", "label"]).copy()
    X = X.dropna(axis=0)
    y = training_data_df["label"].copy()

    # ======= II. Train a quick Random Forest to get features importance =======
    random_forest = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=69)
    random_forest.fit(X, y)

    # ======= III. Get the feature importance from the Random Forest =======
    feature_importance = random_forest.feature_importances_
    importance_df = pd.DataFrame({"feature": X.columns, "importance": feature_importance})

    importance_df = importance_df.sort_values(by="importance", ascending=False)
    top_features = importance_df.head(n_components)["feature"].tolist()

    # ======= IV. Create a new DataFrame with the selected features =======
    # IV.1 Keep only the selected features
    filtered_training_data = X[top_features].copy()

    # IV.2 Add the asset and label columns and drop NaN values from lagged features
    filtered_training_data["asset"] = training_data_df["asset"]
    filtered_training_data["label"] = training_data_df["label"]

    # IV.3 Drop NaN values
    filtered_training_data = filtered_training_data.dropna(axis=0)

    return filtered_training_data


# -----------------------------------------------------------------------------
def descent_selection(training_data_df: pd.DataFrame, n_components: int):
    # ======= I. Initialize the input =======
    filtered_training_data = training_data_df.copy()
    nb_features = filtered_training_data.drop(columns=["asset", "label"]).shape[1]

    # ======= II. Reduce features dimensionality iteratively =======
    while nb_features > (round(n_components * 1.5)):
        # ------- 1. Perform PCA to sort the features by importance -------
        nb_features = nb_features // 2
        filtered_training_data = pca_selection(filtered_training_data, nb_features)

        # ------- 2. Perform Random Forest to sort the features by importance -------
        if nb_features > (round(n_components * 1.5)):
            nb_features = nb_features // 2
            filtered_training_data = rfImportance_selection(filtered_training_data, nb_features)

    return filtered_training_data


# -----------------------------------------------------------------------------
def clustering_trainingSet(priceSeries_df: pd.DataFrame, training_dfs_list: list, last_date: pd.Timestamp, max_length: int):
    def split_sublists(list_of_lists, max_length=4):
        result = []
        for sublist in list_of_lists:
            # Split the sublist into chunks of max_length
            for i in range(0, len(sublist), max_length):
                result.append(sublist[i : i + max_length])

        return result

    priceSeries_df_Cut = priceSeries_df.loc[:last_date].copy()
    clusters_list = clusterizers.riskfolio_clustering(df=priceSeries_df_Cut, linkage="ward")

    training_clusters_list = []
    for cluster in clusters_list:
        assets = cluster.columns.tolist()

        asset_dfs_list = []
        for training_df in training_dfs_list:
            if training_df.columns[0] in assets:
                asset_dfs_list.append(training_df)

        training_clusters_list.append(asset_dfs_list)

    training_clusters_list = split_sublists(training_clusters_list, max_length=max_length)
    clusters_name = [[asset.columns[0] for asset in cluster] for cluster in training_clusters_list]

    return training_clusters_list, clusters_name
