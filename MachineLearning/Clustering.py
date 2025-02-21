"""
# Description: This file contains the functions used to clusterize a given universe of assets.
                -> We expect the input to be a DataFrame containing the prices of the assets and the output to be a list of DataFrames, each containing the assets of a cluster.
                -> REMINDER: We model the spread of a combination of assets as a linear combination of the assets' log(prices).
_____
riskfolio_clustering: uses riskfolio-lib to perform the clustering based on Hierarchical Risk Parity.
dtw_clustering: uses TimeSeriesKMeans from tslearn to perform the clustering based on Dynamic Time Warping.
_____
POTENTIAL IMPROVEMENTS:
    - Add more clustering methods.
    - Specifically design a clustering method to detect co-integrated assets (custom criterion in hierarchical clustering?).
"""

import pandas as pd
import numpy as np
import riskfolio as rp
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import warnings

warnings.filterwarnings("ignore")


# ========================================================================================================== #
def riskfolio_clustering(df: pd.DataFrame, linkage: str):
    """
    Use Riskfolio Library to clusterize assets based on their log prices.

    Args:
        df (pd.DataFrame): DataFrame containing the prices of the assets.
        linkage (str): Linkage method for clustering.
                -> linkages = ['single','complete','average','weighted','centroid', 'median', 'ward','DBHT']

    Returns:
        clusters (list): List containing the DataFrames of assets for each cluster (prices are raw i.e without any transformation).
    """
    # ======== I. Apply log transformation to the prices ======== (As we further model the spread as a linear combination of the assets' log prices)
    log_prices = np.log(df)
    log_prices.dropna(axis=0, inplace=False)

    # ======== II. Performing the clusterization ========
    clusters = rp.assets_clusters(
        returns=log_prices,
        codependence="pearson",
        linkage=linkage,
        k=None,
        max_k=10,
        leaf_order=True,
    )

    # ======== III. Preparing data before returning
    clusters_df = pd.DataFrame(clusters)
    clusters_df.reset_index(drop=True, inplace=True)

    # --------
    liste = {}
    for index, cluster in clusters_df["Clusters"].items():
        if cluster not in liste:
            liste[cluster] = [clusters_df["Assets"][index]]
        else:
            liste[cluster].append(clusters_df["Assets"][index])

    clusters_list = []

    # --------
    for _, tickers in liste.items():
        colonnes_cluster = [ticker for ticker in tickers if ticker in df.columns]
        df_cluster = df[colonnes_cluster]
        clusters_list.append(df_cluster)

    return clusters_list


# ---------------------------------------------------------------------------------------------------------- #
def dtw_clustering(df: pd.DataFrame, n_clusters: int):
    """
    Use TimeSeriesKMeans with Distance Time Warping metric from tslearn to clusterize assets based on their log prices.

    Args:
        df (pd.DataFrame): DataFrame containing the prices of the assets.
        n-clusters (int): represents the number of clusters we want to generate.

    Returns:
        dataframes_clusters (list) -> each dataframe in this list corresponds to the assets of a cluster(prices are raw).
    """
    # ======== I. Apply log transformation to the prices ======== (As we further model the spread as a linear combination of the assets' log prices)
    log_df = np.log(df)
    time_series_data = log_df.T.values

    # ======== II. Scale the time series data ========
    scaler = TimeSeriesScalerMeanVariance()
    time_series_data = scaler.fit_transform(time_series_data)

    # ======== III. Apply TimeSeriesKMeans clustering with DTW metric ========
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
    labels = model.fit_predict(time_series_data)

    # ======== IV. Create a list to hold the resulting dataframes for each cluster ========
    clusters_list = []

    for cluster in range(n_clusters):
        # Select columns belonging to the current cluster
        cluster_columns = df.columns[labels == cluster]
        clusters_list.append(df[cluster_columns])

    return clusters_list
