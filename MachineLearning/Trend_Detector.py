import os
import sys

sys.path.append(os.path.abspath("../"))
import models.Labeller as labeller
import models.Meta_Features as meta_features
import models.Meta_Labeller as meta_labeller

import pandas as pd
import numpy as np
from scipy.stats import loguniform
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import riskfolio as rp

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


class Roseau:

    def __init__(
        self,
        data_df: pd.DataFrame,
        individual_dfs_list: list,
    ):
        """
        The class is used to store the different dataframes and models used in the process of training and testing the model.

        Args:
            data_df (pd.DataFrame): The dataframe containing price data for multiple assets.
            features_params (dict): The parameters to pass to the features methods.
            labelling_params (dict): The parameters to pass to the labelling

        Parameters template:
            -> See the parameters used when the class is initialized with None values.
        """
        # ======= I. Labelled Data =======
        self.data_df = data_df
        self.individual_dfs_list = individual_dfs_list

        self.prepared_dfs_list = None
        self.training_end_index = None

        self.tscv = TimeSeriesSplit(n_splits=3, gap=20)

        # ======= II. Data Splitting (asset-wise then put inside a list) =======
        self.training_data_dfs_list = None
        self.validation_data_dfs_list = None
        self.testing_data_dfs_list = None
        self.production_data_dfs_list = None
        self.covid_data_dfs_list = None

        # ======= III. Training Data Preparation (aggregation) =======
        self.training_data_list = None
        self.clusters_assets = None
        # => We group assets by clusters and then aggregate the data for each cluster. self.training_data_list saves the aggregated data for each cluster.

        self.filtered_training_data_df = None
        self.binary_up_training_data_df = None
        self.binary_down_training_data_df = None
        # => We filter the training data to keep only the most important features and then we create binary labels. Only the cluster used for training is saved here.

        self.meta_training_data_df = None
        # => We create meta labels for the training data to use them in the secondary models.

        # ======= IV. Primary ML Models  =======
        self.primary_random_forest = None
        self.primary_logreg_up = None
        self.primary_logreg_down = None
        self.primary_svc_up = None
        self.primary_svc_down = None
        self.primary_mlp = None

        self.primary_features_used = None
        # => We save the features used to train the model so we can use the same features for the testing data.

        # ======= V. Secondary ML Models (Meta Labelling) =======
        self.secondary_random_forest = None
        self.secondary_logreg_up = None
        self.secondary_logreg_down = None
        self.secondary_svc_up = None
        self.secondary_svc_down = None
        self.secondary_mlp = None

        self.secondary_features_used = None

    # ==========================================================
    # ==================== DATA PREPARATION ====================
    def apply_labelling(
        self,
        labelling_method: str,
        labelling_params: dict,
        individual_dfs_list: list,
    ):
        """
        This method applies the labelling method to the individual dataframes.
        Also save the labeled dataframes in the instance variable labeled_dfs_list which is a list of dataframes.

        Args:
            labelling_method (str): The labelling method to apply.
            nb_assets (int): The number of assets to apply the labelling method to (to manage computation time).
            individual_dfs_list (list): The list of individual dataframes with featured assets.

        Returns:
            labeled_dfs_list (list): The list of labeled dataframes, each dataframe now contain 2 columns : the asset price and the label.
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
                _, _, labeled_series = labeller.combination_labeller(price_series=asset_df[asset_name], params=labelling_params)
                labeled_df["label"] = labeled_series

            elif labelling_method == "regR2rank":
                _, labeled_series, _ = labeller.combination_labeller(price_series=asset_df[asset_name], params=labelling_params)
                labeled_df["label"] = labeled_series

            elif labelling_method == "lookForward":
                labeled_series, _, _ = labeller.combination_labeller(price_series=asset_df[asset_name], params=labelling_params)
                labeled_df["label"] = labeled_series

            else:
                raise ValueError("Invalid labelling method")

            # II.3. Save the labeled dataframe
            labeled_dfs_list.append(labeled_df)

        # ======= III. Save the results =======
        self.prepared_dfs_list = labeled_dfs_list

        return labeled_dfs_list

    # -----------------------------------------------------------
    def meta_labelling(self, primary_model_name: str, labeller: str, validation_data_df: pd.DataFrame):
        """
        This function performs meta-labelling on the validation data using the primary model's predictions.

        Args:
            primary_model_name (str): The name of the primary model to use.
            validation_data_df (pd.DataFrame): The validation data to use.

        Returns:
            secondary_training_data_prepared (pd.DataFrame): The prepared data for training the secondary model.
            secondary_training_data_raw (pd.DataFrame): The raw data for further analysis.
        """
        # ======= I. Make predictions =======
        validation_data = validation_data_df.copy()
        results, _ = self.test_primary_model(model_name=primary_model_name, testing_data_df=validation_data)

        # II.2 Keep only the selected features, asset and label columns & add predictions and meta label
        secondary_training_data = validation_data.copy()
        secondary_training_data["predictions"] = results["predictions"]
        secondary_training_data["meta_label"] = 0

        # ======= III. Meta-labelling =======
        # ------- Binary Meta Labelling -------
        if labeller == "right_wrong":
            meta_training_data = meta_labeller.right_wrong(secondary_training_data)
        elif labeller == "right_wrong_noZero":
            meta_training_data = meta_labeller.right_wrong_noZero(secondary_training_data)
        elif labeller == "trade_lock":
            meta_training_data = meta_labeller.trade_lock(secondary_training_data)

        # ------- Trinary Meta Labelling -------
        elif labeller == "good_bad_ugly":
            meta_training_data = meta_labeller.good_bad_ugly(secondary_training_data)
        elif labeller == "good_bad_ugly_noZero":
            meta_training_data = meta_labeller.good_bad_ugly_noZero(secondary_training_data)
        elif labeller == "gbu_extended":
            meta_training_data = meta_labeller.gbu_extended(secondary_training_data)
        elif labeller == "gbu_extended_noZero":
            meta_training_data = meta_labeller.gbu_extended_noZero(secondary_training_data)

        # ======= IV. Drop the unnecessary columns =======
        meta_training_data = meta_training_data.dropna()

        return meta_training_data

    # -----------------------------------------------------------
    def apply_meta_features(self, meta_dfs_list: list, meta_features_params: dict):
        """
        This method applies the meta features to the meta labeled dataframes.
        The market features should be already computed when training the primary model, so no need to recomputed them. It implies that no features cleaning is needed.

        Args:
            meta_dfs_list (list): The list of meta labeled dataframes.

        Returns:
            featured_meta_dfs_list (list): The list of featured meta dataframes.
        """
        # ======= I. Initialize the input and output =======
        meta_labeled_dfs = meta_dfs_list.copy()
        featured_meta_dfs_list = []

        # ======= II. Apply the features =======
        for asset_df in meta_labeled_dfs:
            featured_df = asset_df.copy()
            predictions_series = featured_df["predictions"].copy()

            # II.1 Compute the features
            for rolling_window in meta_features_params["rolling_windows"]:
                # Average Predictions
                rolling_avg_predictions = meta_features.average_predictions_features(predictions_series=predictions_series, window=rolling_window)
                featured_df[f"rolling_avg_predictions_{rolling_window}"] = rolling_avg_predictions

                # Predictions Volatility
                rolling_predictions_vol = meta_features.volatility_predictions_features(predictions_series=predictions_series, window=rolling_window)
                featured_df[f"rolling_predictions_vol_{rolling_window}"] = rolling_predictions_vol

                # Predictions Changes
                rolling_predictions_changes = meta_features.predictions_changes_features(predictions_series=predictions_series, window=rolling_window)
                featured_df[f"rolling_predictions_changes_{rolling_window}"] = rolling_predictions_changes

                # Predictions Entropy
                if rolling_window < 50:
                    rolling_predictions_shannon, rolling_predictions_plugin, rolling_predictions_lempel_ziv, rolling_predictions_kontoyiannis = meta_features.entropy_predictions_features(predictions_series=predictions_series, window=rolling_window)
                    featured_df[f"rolling_predictions_shannon_{rolling_window}"] = rolling_predictions_shannon
                    featured_df[f"rolling_predictions_plugin_{rolling_window}"] = rolling_predictions_plugin
                    featured_df[f"rolling_predictions_lempel_ziv_{rolling_window}"] = rolling_predictions_lempel_ziv
                    featured_df[f"rolling_predictions_kontoyiannis_{rolling_window}"] = rolling_predictions_kontoyiannis

            featured_df = featured_df.dropna(axis=0)
            featured_meta_dfs_list.append(featured_df)

        return featured_meta_dfs_list

    # ================================================================
    # ==================== TRAINING PREPROCESSING ====================
    def split_samples(
        self,
        training_proportion: float,
        validation_proportion: float,
        dfs_list: list,
    ):
        """
        This method splits the data into training, testing and embargo data.
        It saves the split data in the instance variables training_data_dfs_list, validation_data_dfs_list and testing_data_dfs_list.

        Args:
            training_proportion (float): The proportion of the data to use for training.
            validation_proportion (float): The proportion of the data to use for validation.
            featured_dfs_list (list): The list of featured dataframes.

        Returns:
            training_data_dfs_list (list): The list of training dataframes.
            validation_data_dfs_list (list): The list of validation dataframes.
            testing_data_dfs_list (list): The list of testing dataframes.
        """
        # ======= I. Initialization of input and output =======
        dfs = dfs_list.copy()
        training_data_dfs_list = []
        validation_data_dfs_list = []
        testing_data_dfs_list = []

        # ======= II. Compute the index to split the data =======
        # II.1 Use the biggest asset to compute the size of the data
        biggest_size = 0
        for asset_df in dfs:
            size_data = len(asset_df)
            if size_data > biggest_size:
                biggest_size = size_data
                biggest_asset = asset_df

        # II.1 Training data indexes
        training_start_index = biggest_asset.index[0]
        training_end_index = biggest_asset.index[int(training_proportion * size_data)]

        # II.2 Validation data indexes
        validation_start_index = biggest_asset.index[int(training_proportion * size_data + 1)]
        validation_end_index = biggest_asset.index[int((validation_proportion + training_proportion) * size_data - 1)]

        # II.3 Testing data indexes
        if validation_proportion + training_proportion != 1:
            testing_start_index = biggest_asset.index[int((validation_proportion + training_proportion) * size_data + 1)]
            testing_end_index = biggest_asset.index[-1]
        else:
            testing_data_dfs_list = None

        # ======= III. Split the data =======
        for asset_df in dfs:
            # III.1 Training data
            training_data = asset_df.loc[training_start_index:training_end_index]
            training_data.dropna(axis=0)
            if len(training_data) > 50:
                training_data_dfs_list.append(training_data)

            # III.2 Validation data
            validation_data = asset_df.loc[validation_start_index:validation_end_index]
            validation_data.dropna(axis=0)
            if len(validation_data) > 50:
                validation_data_dfs_list.append(validation_data)

            # III.3 Testing data
            if validation_proportion + training_proportion < 1:
                testing_data = asset_df.loc[testing_start_index:testing_end_index]
                testing_data.dropna(axis=0)
                if len(testing_data) > 50:
                    testing_data_dfs_list.append(testing_data)

        return training_data_dfs_list, validation_data_dfs_list, testing_data_dfs_list

    # -----------------------------------------------------------
    def split_samples_cartesius(
        self,
        prepared_dfs_list: list = None,
    ):
        # ======= 0. Class instances management =======
        if prepared_dfs_list is not None:
            self.prepared_dfs_list = prepared_dfs_list

        # ======= I. Initialization of input and output =======
        prepared_dfs = self.prepared_dfs_list.copy()
        training_data_dfs_list = []
        validation_data_dfs_list = []
        testing_data_dfs_list = []
        production_data_dfs_list = []
        covid_data_dfs_list = []

        # ======= II. Compute the index to split the data =======
        # II.1 Training data indexes
        training_start_index = "2011-01-01"
        training_end_index = "2017-01-01"
        self.training_end_index = training_end_index

        # II.2 Validation data indexes
        validation_start_index = "2017-01-01"
        validation_end_index = "2021-07-01"

        # II.3 Testing data indexes
        testing_start_index = "2021-07-01"
        testing_end_index = "2023-04-01"

        # II.4 Production data indexes
        production_start_index = "2023-04-01"
        production_end_index = "2024-10-31"

        # II.5 Covid data indexes
        covid_start_index = "2020-02-01"
        covid_end_index = "2020-06-30"

        # ======= III. Split the data =======
        for asset_df in prepared_dfs:
            # III.1 Training data
            training_data = asset_df.loc[training_start_index:training_end_index]
            training_data.dropna(axis=0)
            if len(training_data) > 50:
                training_data_dfs_list.append(training_data)

            # III.2 Validation data
            validation_data = asset_df.loc[validation_start_index:validation_end_index]
            validation_data.dropna(axis=0)
            if len(validation_data) > 50:
                validation_data_dfs_list.append(validation_data)

            # III.3 Testing data
            testing_data = asset_df.loc[testing_start_index:testing_end_index]
            testing_data.dropna(axis=0)
            if len(testing_data) > 50:
                testing_data_dfs_list.append(testing_data)

            # III.4 Production data
            production_data = asset_df.loc[production_start_index:production_end_index]
            production_data.dropna(axis=0)
            if len(production_data) > 50:
                production_data_dfs_list.append(production_data)

            # III.5 Covid data
            covid_data = asset_df.loc[covid_start_index:covid_end_index]
            covid_data.dropna(axis=0)
            if len(covid_data) > 50:
                covid_data_dfs_list.append(covid_data)

        # ======= IV. Save the splitted data =======
        self.training_data_dfs_list = training_data_dfs_list
        self.validation_data_dfs_list = validation_data_dfs_list
        self.testing_data_dfs_list = testing_data_dfs_list
        self.production_data_dfs_list = production_data_dfs_list
        self.covid_data_dfs_list = covid_data_dfs_list

        return (
            training_data_dfs_list,
            validation_data_dfs_list,
            testing_data_dfs_list,
            production_data_dfs_list,
            covid_data_dfs_list,
        )

    # -----------------------------------------------------------
    def aggregate_training_data(self, training_data_dfs_list: list, rebalancing_strategy: str):
        """
        This method aggregates the training data into a single dataframe in order to train the model on data from multiple assets (more data to train + more generalization).
        The groups of assets are determined by clustering the assets based on their price series and limiting the number of assets in each group to 4.

        Args:
            training_data_dfs_list (list): The list of training dataframes to aggregate (they should have the same columns which are the asset price + label + features).

        Returns:
            training_data (pd.DataFrame): The aggregated training data.
        """
        # ======= I. Initialize input and output =======
        training_data_dfs = training_data_dfs_list.copy()
        training_data = pd.DataFrame()

        # ======= II. Clustering to aggregate similar data =======
        training_df = self.data_df.copy()
        training_df = training_df.loc[: self.training_end_index].dropna(axis=1)
        clusters_list = riskfolio_clustering(df=training_df, linkage="ward")

        # II.1 Split the clusters into sub-clusters of at most 4 assets each
        new_clusters_list = []
        for cluster in clusters_list:
            if cluster.shape[1] > 4:
                for i in range(0, cluster.shape[1], 4):
                    sub_cluster = cluster.iloc[:, i : i + 4]
                    new_clusters_list.append(sub_cluster)
            else:
                new_clusters_list.append(cluster)

        # ======= III. Aggregate the data for each cluster =======
        training_data_list = []
        clusters_assets = []
        for cluster in new_clusters_list:
            training_data = pd.DataFrame()
            list_assets = cluster.columns.tolist()
            clusters_assets.append(list_assets)

            for asset_df in training_data_dfs:
                if asset_df.columns[0] in list_assets:
                    renamed_df = asset_df.copy()
                    renamed_df.rename(columns={renamed_df.columns[0]: "asset"}, inplace=True)
                    training_data = pd.concat([training_data, renamed_df], axis=0, ignore_index=True)

            if rebalancing_strategy is not None:
                balanced_training_data = self.rebalance_classes(training_data=training_data, strategy=rebalancing_strategy)
            else:
                balanced_training_data = training_data
            balanced_training_data = balanced_training_data.dropna(axis=0)
            training_data_list.append(balanced_training_data)

        # ======= V. Save the results =======
        self.training_data_list = training_data_list
        self.clusters_assets = clusters_assets

        return training_data_list, clusters_assets

    # -----------------------------------------------------------
    def aggregate_meta_data(self, secondary_data_dfs_list: list, rebalancing_strategy: str):
        """
        This method aggregates the meta training data into a single dataframe in order to train the secondary model on data from multiple assets (more data to train + more generalization).
        The labels are replaced by the meta labels which are 1 if the prediction is a True Positive and 0 otherwise.

        Args:
            training_data_dfs_list (list): The list of training dataframes to aggregate (they should have the same columns which are the asset price + label + features).

        Returns:
            training_data (pd.DataFrame): The aggregated training data.
        """
        # ======= I. Initialize input and output =======
        training_data_dfs = secondary_data_dfs_list.copy()
        training_data = pd.DataFrame()

        # ======= II. Aggregate the data =======
        for asset_df in training_data_dfs:
            renamed_df = asset_df.copy()
            renamed_df = renamed_df.drop(columns=["label"])
            renamed_df.rename(columns={"meta_label": "label"}, inplace=True)
            renamed_df.rename(columns={renamed_df.columns[0]: "asset"}, inplace=True)
            training_data = pd.concat([training_data, renamed_df], axis=0, ignore_index=True)

        if rebalancing_strategy is not None:
            balanced_training_data = self.rebalance_classes(training_data=training_data, strategy=rebalancing_strategy)
        else:
            balanced_training_data = training_data
        balanced_training_data = balanced_training_data.dropna(axis=0)

        # ======= III. Save the results =======
        self.meta_training_data_df = balanced_training_data

        return balanced_training_data

    # -----------------------------------------------------------
    def rebalance_classes(self, training_data: pd.DataFrame, strategy: str):
        """
        Rebalance the classes in the 'label' column of the DataFrame.

        Parameters:
            training_data (pd.DataFrame): Input DataFrame with 'label' and feature columns.
            strategy (str): Rebalancing strategy, options:
                - "oversample" (default) -> Random Over Sampling
                - "undersample" -> Random Under Sampling
                - "smote" -> Synthetic Minority Over-sampling Technique

        Returns:
            pd.DataFrame: Rebalanced DataFrame with equal class distribution.
        """
        X = training_data.drop(columns=["label"]).copy()
        y = training_data["label"].copy()

        if strategy == "oversample":
            sampler = RandomOverSampler(random_state=69)
        elif strategy == "undersample":
            sampler = RandomUnderSampler(random_state=69)
        elif strategy == "smoteenn":
            sampler = SMOTEENN(random_state=69)
        else:
            raise ValueError("Invalid strategy. Choose from 'oversample', 'undersample', or 'smoteenn'.")

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        resampled_data = pd.DataFrame(X_resampled, columns=X_resampled.columns).assign(label=y_resampled)

        return resampled_data

    # -----------------------------------------------------------
    def clean_features(self, n_components: int, training_data_df: pd.DataFrame = None):
        """
        This method cleans the features by performing PCA and selecting the most important ones, they are then aligned on multiple days.
        The hypothesis behind this method is that the most important features are the most stable and therefore the most useful, the lagged features are also added to capture the temporal dependencies.

        We also creates the binary sets for the Logistic Regression model which is two dataframes with the labels transformed to keep only one side (either upper_trends/constant or lower_trends/constant),
        this is done to train two models that will predict the upper and lower trends separately (As the Logistic Regression fit a sigmoid function and we input features mostly with a linear relationship with the label).

        Args:
            n_components (int): The number of components to keep after PCA.
            training_data_df (pd.DataFrame): The training data to use (can be the dataframe of a single asset if the column of teh price series is called 'asset').

        Returns:
            filtered_training_data (pd.DataFrame): The training data with the selected features.
        """
        # ======= 0. Class instances management =======
        if training_data_df is not None:
            self.training_data_df = training_data_df

        # ======= I. Initialize the input =======
        training_data = self.training_data_df.copy()
        X = training_data.drop(columns=["asset", "label"]).copy()
        X = X.dropna(axis=0)
        y = training_data["label"].copy()

        # ======= II. Train a quick Random Forest to get features importance =======
        # II.1 Train the model
        random_forest = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=69)
        random_forest.fit(X, y)

        # II.2 Get the feature importance
        rf_feature_importance = random_forest.feature_importances_
        rf_importance_df = pd.DataFrame({"feature": X.columns, "importance": rf_feature_importance})

        # II.3 Sort the features by importance
        rf_importance_df = rf_importance_df.sort_values(by="importance", ascending=False)
        rf_top_features = rf_importance_df.head(2 * n_components)["feature"].tolist()

        # ======= III. Perform PCA to sort the features by importance =======
        # III.1 Standardize the data because PCA is sensitive to the scale of the features
        scaler = StandardScaler()
        X_standardized = X[rf_top_features].copy()
        X_standardized = pd.DataFrame(scaler.fit_transform(X_standardized), columns=X_standardized.columns)

        # III.2 Perform PCA
        pca = PCA(n_components=n_components, random_state=69)
        pca.fit(X_standardized)

        # III.3 Get the loadings and feature importance
        loadings = np.abs(pca.components_)
        pca_feature_importance = loadings.sum(axis=0)
        pca_importance_df = pd.DataFrame({"feature": X_standardized.columns, "importance": pca_feature_importance})

        # III.4 Sort the features by importance
        pca_importance_df = pca_importance_df.sort_values(by="importance", ascending=False)
        pca_top_features = pca_importance_df.head(n_components)["feature"].tolist()

        # ======= IV. Keep the features that are both in RF and PCA top features =======
        top_features = list(set(rf_top_features) & set(pca_top_features))

        # ======= IV. Create a new DataFrame with the selected features =======
        # IV.1 Keep only the selected features
        filtered_training_data = X[top_features].copy()

        features_used = filtered_training_data.columns
        self.features_used = features_used

        # IV.2 Add the asset and label columns and drop NaN values from lagged features
        filtered_training_data["asset"] = training_data["asset"]
        filtered_training_data["label"] = training_data["label"]
        filtered_training_data = filtered_training_data.dropna(axis=0)

        # ======= V. Create the binary set for Logistic Regression =======
        # V.1 Create the binary sets
        binary_up_training_data = filtered_training_data.copy()
        binary_down_training_data = filtered_training_data.copy()

        # V.2 Transform the labels to keep only one side (either upper_trends/constant or lower_trends/constant)
        binary_up_training_data["label"] = binary_up_training_data["label"].apply(lambda x: 1 if x == 1 else 0)
        binary_down_training_data["label"] = binary_down_training_data["label"].apply(lambda x: -1 if x == -1 else 0)

        # ======= VI. Save the dataframes =======
        self.filtered_training_data_df = filtered_training_data
        self.binary_up_training_data_df = binary_up_training_data
        self.binary_down_training_data_df = binary_down_training_data

        return (
            filtered_training_data,
            binary_up_training_data,
            binary_down_training_data,
        )

    # ========================================================
    # ==================== LEARNING LOGIC ====================
    def train_RandomForestClassifier(
        self,
        n_jobs: int,
        max_comb: int,
        filtered_training_data_df: pd.DataFrame,
        model_order: str,
    ):
        """
        This method trains a Random Forest Classifier on the training data with more hyperparameters
        and provides insights on the model.
        Also saves the model in the instance variable `self.random_forest`.

        Args:
            filtered_training_data_df (pd.DataFrame): The training data to use.

        Returns:
            random_forest (RandomForestClassifier): The trained model.
            train_metrics (dict): Dictionary containing metrics and insights about the trained model.
        """
        # ======= 0. Initialize Training Data =======
        training_data = filtered_training_data_df.copy()

        # ======= I. Initialize Model and Parameters =======
        param_dist = {"n_estimators": [100, 150, 200, 300, 500, 1000], "max_depth": [2, 3, 4, 5], "min_samples_split": [10, 15, 20, 25, 30], "min_samples_leaf": [5, 10, 15, 20, 30], "max_features": ["auto", "sqrt", "log2"], "class_weight": ["balanced", {-1: 2, 0: 1, 1: 1}, {-1: 1, 0: 2, 1: 1}, {-1: 1, 0: 1, 1: 2}]}

        random_forest = RandomForestClassifier(random_state=69)

        randomized_search = RandomizedSearchCV(
            estimator=random_forest,
            param_distributions=param_dist,
            n_iter=max_comb,  # Number of random combinations to try
            cv=self.tscv,  # k-fold cross-validation
            scoring="f1_macro",
            verbose=0,
            n_jobs=n_jobs,
            random_state=69,
        )

        # ======= II. Train the model =======
        try:
            X = training_data.drop(columns=["asset", "label"]).copy()
            y = training_data["label"].copy()

            if X.isnull().values.any() or y.isnull().values.any():
                raise ValueError("Training data contains missing values.")

            randomized_search.fit(X, y)
        except Exception as e:
            print(f"Error training model: {e}")
            raise

        # ======= III. Save the best model =======
        random_forest = randomized_search.best_estimator_
        features_used = X.columns
        if model_order == "primary":
            self.primary_random_forest = random_forest
            self.primary_features_used = features_used
        elif model_order == "secondary":
            self.secondary_random_forest = random_forest
            self.secondary_features_used = features_used
        else:
            raise ValueError("Model order should be either 'primary' or 'secondary'.")

        # ======= IV. Provide insights about the model =======
        feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": random_forest.feature_importances_}).sort_values(by="Importance", ascending=False)

        training_accuracy = random_forest.score(X, y)
        cv_score = randomized_search.best_score_
        class_distribution = y.value_counts(normalize=True).sort_index()

        tree_depths = [estimator.tree_.max_depth for estimator in random_forest.estimators_]
        avg_tree_depth = sum(tree_depths) / len(tree_depths)

        total_nodes = [estimator.tree_.node_count for estimator in random_forest.estimators_]
        avg_nodes = sum(total_nodes) / len(total_nodes)

        train_metrics = {
            "feature_importances": feature_importances,
            "training_accuracy": training_accuracy,
            "cv_score": cv_score,
            "class_distribution": class_distribution,
            "avg_tree_depth": avg_tree_depth,
            "avg_nodes": avg_nodes,
            "parameters": randomized_search.best_params_,
        }

        return random_forest, train_metrics

    # -----------------------------------------------------------
    def train_LogisticRegression(
        self,
        n_jobs: int,
        max_comb: int,
        binary_up_training_data_df: pd.DataFrame,
        binary_down_training_data_df: pd.DataFrame,
        model_order: str,
    ):
        """
        This method trains two Logistic Regression models (one for upside trends and one for downside trends) using RandomizedSearchCV
        for hyperparameter optimization and provides insights on the model.

        Args:
            binary_up_training_data_df (pd.DataFrame): The training data to use for the upside trends.
            binary_down_training_data_df (pd.DataFrame): The training data to use for the downside trends.

        Returns:
            logistic_regression_up (LogisticRegression): The trained model for the upside trends.
            logistic_regression_down (LogisticRegression): The trained model for the downside trends.
            train_metrics (dict): The insights about the models.
        """
        # ======= 0. Initialize Training Data =======
        training_data_up = binary_up_training_data_df.copy()
        training_data_down = binary_down_training_data_df.copy()

        # ======= I. Initialize Model and Parameters =======
        param_dist_up = {
            "C": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 1000],
            "max_iter": [100, 200, 500, 1000, 2000],
            "class_weight": [{0: 1, 1: 1}, {0: 0.5, 1: 1}, {0: 1, 1: 0.5}, {0: 1, 1: 2}, {0: 2, 1: 1}],
        }
        param_dist_down = {
            "C": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 1000],
            "max_iter": [100, 200, 500, 1000, 2000],
            "class_weight": [{0: 1, -1: 1}, {0: 0.5, -1: 1}, {0: 1, -1: 0.5}, {0: 1, -1: 2}, {0: 2, -1: 1}],
        }

        logistic_regression_up = LogisticRegression(solver="saga", random_state=69)
        logistic_regression_down = LogisticRegression(solver="saga", random_state=69)

        # ======= II. Randomized Search for Logistic Regression Models =======
        # RandomizedSearch for upside trend model
        random_search_up = RandomizedSearchCV(
            estimator=logistic_regression_up,
            param_distributions=param_dist_up,
            n_iter=max_comb,  # Number of random combinations to try
            cv=self.tscv,  # k-fold cross-validation
            scoring="f1_macro",
            verbose=0,
            random_state=69,
            n_jobs=n_jobs,
        )

        # RandomizedSearch for downside trend model
        random_search_down = RandomizedSearchCV(
            estimator=logistic_regression_down,
            param_distributions=param_dist_down,
            n_iter=max_comb,  # Number of random combinations to try
            cv=self.tscv,  # k-fold cross-validation
            scoring="f1_macro",
            verbose=0,
            random_state=69,
            n_jobs=n_jobs,
        )

        # ======= III. Train the models using Randomized Search =======
        try:
            # Preparing the data
            X_up = training_data_up.drop(columns=["asset", "label"])
            y_up = training_data_up["label"]
            X_down = training_data_down.drop(columns=["asset", "label"])
            y_down = training_data_down["label"]

            if X_up.isnull().values.any() or y_up.isnull().values.any():
                raise ValueError("Training data contains missing values.")

            if X_down.isnull().values.any() or y_down.isnull().values.any():
                raise ValueError("Training data contains missing values.")

            # Fitting Randomized Search for both models
            random_search_up.fit(X_up, y_up)
            random_search_down.fit(X_down, y_down)

        except Exception as e:
            print(f"Error training model: {e}")
            raise

        # ======= IV. Save the best models =======
        logistic_regression_up = random_search_up.best_estimator_
        logistic_regression_down = random_search_down.best_estimator_

        features_used = X_up.columns

        if model_order == "primary":
            self.primary_logreg_up = logistic_regression_up
            self.primary_logreg_down = logistic_regression_down
            self.primary_features_used = features_used
        elif model_order == "secondary":
            self.secondary_logreg_up = logistic_regression_up
            self.secondary_logreg_down = logistic_regression_down
            self.secondary_features_used = features_used
        else:
            raise ValueError("Model order should be either 'primary' or 'secondary'.")

        # ======= V. Provide insights about the model =======
        feature_importances_up = pd.DataFrame({"Feature": X_up.columns, "Coefficient": logistic_regression_up.coef_[0]}).sort_values(by="Coefficient", ascending=False)
        training_accuracy_up = logistic_regression_up.score(X_up, y_up)
        cv_score_up = random_search_up.best_score_
        class_distribution_up = y_up.value_counts(normalize=True).sort_index()

        feature_importances_down = pd.DataFrame({"Feature": X_down.columns, "Coefficient": logistic_regression_down.coef_[0]}).sort_values(by="Coefficient", ascending=False)
        training_accuracy_down = logistic_regression_down.score(X_down, y_down)
        cv_score_down = random_search_down.best_score_
        class_distribution_down = y_down.value_counts(normalize=True).sort_index()

        train_metrics = {
            "feature_importances": [feature_importances_up, feature_importances_down],
            "training_accuracy": [training_accuracy_up, training_accuracy_down],
            "cv_score": [cv_score_up, cv_score_down],
            "class_distribution": [class_distribution_up, class_distribution_down],
            "parameters_up": random_search_up.best_params_,
            "parameters_down": random_search_down.best_params_,
        }

        return logistic_regression_up, logistic_regression_down, train_metrics

    # -----------------------------------------------------------
    def train_SVC(
        self,
        n_jobs: int,
        max_comb: int,
        binary_up_training_data_df: pd.DataFrame,
        binary_down_training_data_df: pd.DataFrame,
        model_order: str,
    ):
        """
        This method trains two Support Vector Classifier models (one for upside trends and one for downside trends)
        using RandomizedSearchCV for hyperparameter optimization and provides insights on the model.

        Args:
            binary_up_training_data_df (pd.DataFrame): The training data to use for the upside trends.
            binary_down_training_data_df (pd.DataFrame): The training data to use for the downside trends.

        Returns:
            svc_up (SVC): The trained model for the upside trends.
            svc_down (SVC): The trained model for the downside trends.
            train_metrics (dict): The insights about the models.
        """
        # ======= 0. Initialize Training Data =======
        training_data_up = binary_up_training_data_df.copy()
        training_data_down = binary_down_training_data_df.copy()

        # ======= I. Initialize Model and Parameters =======
        param_dist = {
            "C": [0.01, 0.1, 1, 2.5, 5, 10, 100, 1000],
            "kernel": ["linear", "rbf", "poly"],
            "max_iter": [500, 1000, 2000, 5000],
        }

        svc_up = SVC(random_state=69, probability=True)
        svc_down = SVC(random_state=69, probability=True)

        # ======= II. Randomized Search for Support Vector Classifiers =======
        random_search_up = RandomizedSearchCV(
            estimator=svc_up,
            param_distributions=param_dist,
            n_iter=max_comb,  # Number of random combinations to try
            cv=self.tscv,  # k-fold cross-validation
            scoring="f1_macro",
            verbose=0,
            random_state=69,
            n_jobs=n_jobs,
        )

        random_search_down = RandomizedSearchCV(
            estimator=svc_down,
            param_distributions=param_dist,
            n_iter=max_comb,  # Number of random combinations to try
            cv=self.tscv,  # k-fold cross-validation
            scoring="f1_macro",
            verbose=0,
            random_state=69,
            n_jobs=n_jobs,
        )

        # ======= III. Train the models using Randomized Search =======
        try:
            # Preparing the data
            X_up = training_data_up.drop(columns=["asset", "label"])
            y_up = training_data_up["label"]
            X_down = training_data_down.drop(columns=["asset", "label"])
            y_down = training_data_down["label"]

            if X_up.isnull().values.any() or y_up.isnull().values.any():
                raise ValueError("Training data contains missing values.")

            if X_down.isnull().values.any() or y_down.isnull().values.any():
                raise ValueError("Training data contains missing values.")

            # Fitting Randomized Search for both models
            random_search_up.fit(X_up, y_up)
            random_search_down.fit(X_down, y_down)

        except Exception as e:
            print(f"Error training model: {e}")
            raise

        # ======= IV. Save the best models =======
        svc_up = random_search_up.best_estimator_
        svc_down = random_search_down.best_estimator_
        features_used = X_up.columns

        if model_order == "primary":
            self.primary_svc_up = svc_up
            self.primary_svc_down = svc_down
            self.primary_features_used = features_used
        elif model_order == "secondary":
            self.secondary_svc_up = svc_up
            self.secondary_svc_down = svc_down
            self.secondary_features_used = features_used
        else:
            raise ValueError("Model order should be either 'primary' or 'secondary'.")

        # ======= V. Provide insights about the model =======
        training_accuracy_up = svc_up.score(X_up, y_up)
        class_distribution_up = y_up.value_counts(normalize=True).sort_index()
        cv_score_up = random_search_up.best_score_

        training_accuracy_down = svc_down.score(X_down, y_down)
        class_distribution_down = y_down.value_counts(normalize=True).sort_index()
        cv_score_down = random_search_down.best_score_

        train_metrics = {
            "training_accuracy": [training_accuracy_up, training_accuracy_down],
            "cv_score": [cv_score_up, cv_score_down],
            "class_distribution": [class_distribution_up, class_distribution_down],
            "parameters_up": random_search_up.best_params_,
            "parameters_down": random_search_down.best_params_,
        }

        return svc_up, svc_down, train_metrics

    # -----------------------------------------------------------
    def train_MLP(
        self,
        n_jobs: int,
        max_comb: int,
        filtered_training_data_df: pd.DataFrame,
        model_order: str,
    ):
        """
        This method trains a Multi-layer Perceptron Classifier using RandomizedSearchCV for hyperparameter optimization
        and provides insights on the model.

        Args:
            filtered_training_data_df (pd.DataFrame): The training data to use.

        Returns:
            mlp (MLPClassifier): The trained model.
            train_metrics (dict): The insights about the model.
        """
        # ======= 0. Initialize Training Data =======
        training_data = filtered_training_data_df.copy()

        # ======= I. Initialize Model and Parameters =======
        param_dist = {
            "hidden_layer_sizes": [(50,), (100,), (150,), (200,), (100, 50)],
            "activation": ["relu", "tanh", "logistic"],
            "solver": ["adam", "sgd"],
            "alpha": loguniform(1e-2, 1e2),
            "learning_rate_init": loguniform(1e-6, 1e-2),
            "max_iter": [100, 200, 300, 500],
        }

        mlp = MLPClassifier(random_state=69, early_stopping=True, n_iter_no_change=10)

        # ======= II. Randomized Search for MLPClassifier =======
        random_search = RandomizedSearchCV(
            estimator=mlp,
            param_distributions=param_dist,
            n_iter=max_comb,  # Number of random combinations to try
            cv=self.tscv,  # k-fold cross-validation
            scoring="f1_macro",
            verbose=0,
            random_state=69,
            n_jobs=n_jobs,
        )

        # ======= III. Train the model using Randomized Search =======
        try:
            X = training_data.drop(columns=["asset", "label"])
            y = training_data["label"]

            if X.isnull().values.any() or y.isnull().values.any():
                raise ValueError("Training data contains missing values.")

            random_search.fit(X, y)

        except Exception as e:
            print(f"Error training model: {e}")
            raise

        # ======= IV. Save the best model =======
        mlp = random_search.best_estimator_
        features_used = X.columns

        if model_order == "primary":
            self.primary_mlp = mlp
            self.primary_features_used = features_used
        elif model_order == "secondary":
            self.secondary_mlp = mlp
            self.secondary_features_used = features_used
        else:
            raise ValueError("Model order should be either 'primary' or 'secondary'.")

        # ======= V. Provide insights about the model =======
        training_accuracy = mlp.score(X, y)
        cv_score = random_search.best_score_
        class_distribution = y.value_counts(normalize=True).sort_index()

        train_metrics = {
            "training_accuracy": training_accuracy,
            "cv_score": cv_score,
            "class_distribution": class_distribution,
            "parameters": random_search.best_params_,
        }

        return mlp, train_metrics

    # ===========================================================
    # ==================== TESTING LOGIC ====================
    def test_primary_model(self, model_name: str, testing_data_df: pd.DataFrame):
        """
        This method tests the model on the testing data.

        Args:
            model_name (str): The name of the model to use.
            testing_data_df (pd.DataFrame): The testing data to use.

        Returns:
            predictions (np.array): The predictions of the model.
        """
        # ======= I. Initialization =======
        testing_data = testing_data_df.copy()
        asset_name = testing_data.columns[0]
        testing_data = testing_data.dropna(axis=0)

        # I.1 Keep only the selected features and the asset and label columns
        kept_columns = [asset_name, "label"] + self.primary_features_used.tolist()
        test_df = testing_data[kept_columns].copy()

        # I.2 Initialize the results dataframe
        results_df = pd.DataFrame()
        results_df[asset_name] = testing_data[asset_name]

        # ======= II. Make the predictions =======
        try:
            X = test_df.drop(columns=[asset_name, "label"])
            y = test_df["label"]

            if X.isnull().values.any() or y.isnull().values.any():
                raise ValueError("Testing data contains missing values.")

            if model_name == "RandomForestClassifier":
                predictions = self.primary_random_forest.predict(X)

            elif model_name == "LogisticRegression":
                predictions_up = self.primary_logreg_up.predict(X)
                predictions_down = self.primary_logreg_down.predict(X)
                predictions = predictions_up + predictions_down

            elif model_name == "SVC":
                predictions_up = self.primary_svc_up.predict(X)
                predictions_down = self.primary_svc_down.predict(X)
                predictions = predictions_up + predictions_down

            elif model_name == "MLP":
                predictions = self.primary_mlp.predict(X)

            else:
                raise ValueError("Model name not recognized.")

        except Exception as e:
            print(f"Error testing model: {e}")
            raise

        # ======= III. Save the predictions =======
        def apply_confirmation(predictions: pd.Series):
            """
            This function applies the confirmation rule to the predictions.

            Args:
                predictions (pd.Series): The predictions to apply the confirmation rule to.

            Returns:
                modified_predictions (pd.Series): The predictions after applying the confirmation rule.
            """
            modified_predictions = predictions.copy()
            for i in range(1, len(predictions)):
                if predictions[i] != modified_predictions[i - 1]:
                    modified_predictions[i] = predictions[i - 1]

            return modified_predictions

        results_df["predictions"] = predictions  # apply_confirmation(predictions)
        results_df["label"] = y

        modified_predictions = predictions.copy()
        for i in range(1, len(predictions) - 1):
            if predictions[i] != predictions[i - 1]:
                modified_predictions[i] = predictions[i - 1]

        # ======= IV. Evaluate the predictions =======
        classification_accuracy = (results_df["predictions"] == results_df["label"]).mean()

        true_positive = ((results_df["predictions"] == 1) & (results_df["label"] == 1)).sum()
        false_positive = ((results_df["predictions"] == 1) & (results_df["label"].isin([0, -1]))).sum() + ((results_df["predictions"] == -1) & (results_df["label"].isin([0, 1]))).sum()
        false_negative = ((results_df["predictions"] == 0) & (results_df["label"].isin([1, -1]))).sum()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        test_metrics = {
            "classification_accuracy": classification_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        return results_df, test_metrics

    # -----------------------------------------------------------
    def test_secondary_model(self, model_name: str, testing_data_df: pd.DataFrame):
        """
        This method tests the model on the testing data.

        Args:
            model_name (str): The name of the model to use.
            testing_data_df (pd.DataFrame): The testing data to use.

        Returns:
            predictions (np.array): The predictions of the model.
        """
        # ======= I. Initialization =======
        testing_data = testing_data_df.copy()
        asset_name = testing_data.columns[0]
        predictions = testing_data["predictions"].copy()

        # I.2 Keep only the selected features and the asset and label columns
        kept_columns = [asset_name, "label"] + self.secondary_features_used.tolist()
        test_df = testing_data[kept_columns].copy()

        # I.3 Initialize the results dataframe
        results_df = pd.DataFrame()
        results_df[asset_name] = testing_data[asset_name]

        # ======= II. Make the predictions =======
        try:
            X = test_df.drop(columns=[asset_name, "label"])
            y = test_df["label"]

            if X.isnull().values.any() or y.isnull().values.any():
                raise ValueError("Testing data contains missing values.")

            if model_name == "RandomForestClassifier":
                meta_predictions = self.secondary_random_forest.predict(X)
                probas = self.secondary_random_forest.predict_proba(X)[:, 1]

            elif model_name == "LogisticRegression":
                meta_predictions_up = self.secondary_logreg_up.predict(X)
                meta_predictions_down = self.secondary_logreg_down.predict(X)
                meta_predictions = meta_predictions_up + meta_predictions_down
                probas = (self.secondary_logreg_up.predict_proba(X)[:, 1] + self.secondary_logreg_down.predict_proba(X)[:, 1]) / 2

            elif model_name == "SVC":
                meta_predictions_up = self.secondary_svc_up.predict(X)
                meta_predictions_down = self.secondary_svc_down.predict(X)
                meta_predictions = meta_predictions_up + meta_predictions_down
                probas = (self.secondary_svc_up.predict_proba(X)[:, 1] + self.secondary_svc_down.predict_proba(X)[:, 1]) / 2

            elif model_name == "MLP":
                meta_predictions = self.secondary_mlp.predict(X)
                probas = self.secondary_mlp.predict_proba(X)[:, 1]

            else:
                raise ValueError("Model name not recognized.")

        except Exception as e:
            print(f"Error testing model: {e}")
            raise

        # ======= III. Save the predictions =======
        results_df["label"] = y
        results_df["predictions"] = predictions
        results_df["meta_predictions"] = meta_predictions
        results_df["probas"] = probas

        # ======= IV. Evaluate the predictions =======
        classification_accuracy = (results_df["meta_predictions"] == results_df["label"]).mean()

        true_positive = ((results_df["meta_predictions"] == 1) & (results_df["label"] == 1)).sum()
        false_positive = ((results_df["meta_predictions"] == 1) & (results_df["label"].isin([0, -1]))).sum() + ((results_df["meta_predictions"] == -1) & (results_df["label"].isin([0, 1]))).sum()
        false_negative = ((results_df["meta_predictions"] == 0) & (results_df["label"].isin([1, -1]))).sum()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        test_metrics = {
            "classification_accuracy": classification_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        return results_df, test_metrics


# ==========================================================
# ==================== HELPER FUNCTIONS ====================
def compute_stats(
    returns: pd.Series,
    market_returns: pd.Series = None,
    risk_free_rate: float = 0.0,
    frequence: str = "daily",
):
    """
    Compute the statistics of the investment.

    Args:
        returns (pd.Series): Series of returns of the investment.
        market_returns (pd.Series): Series of returns of the market index for comparison.
        risk_free_rate (float): Risk-free rate for certain calculations.
        frequence (str): Frequence of the returns.

    Returns:
        stats (dict): Dictionary containing the statistics of the investment, including:

        ======= Returns distribution statistics =======
        - **Expected Return**: The annualized mean return, indicating average performance.
        - **Volatility**: Standard deviation of returns, representing total risk.
        - **Downside Deviation**: Standard deviation of negative returns, used in risk-adjusted metrics like Sortino Ratio.
        - **Median Return**: The median of returns, a measure of central tendency.
        - **Skew** and **Kurtosis**: Describe the distribution shape, with skew indicating asymmetry and kurtosis indicating tail heaviness.

        ======= Risk measures =======
        - **Maximum Drawdown**: Largest observed loss from peak to trough, a measure of downside risk.
        - **Max Drawdown Duration**: Longest period to recover from drawdown, indicating risk recovery time.
        - **VaR 95** and **CVaR 95**: Value at Risk and Conditional Value at Risk at 95%, giving the maximum and average expected losses in worst-case scenarios.

        ======= Market sensitivity measures =======
        - **Beta**: Sensitivity to market movements.
        - **Alpha**: Risk-adjusted return above the market return.
        - **Upside/Downside Capture Ratios**: Percent of market gains or losses captured by the investment.
        - **Tracking Error**: Volatility of return differences from the market.

        ======= Performance measures =======
        - **Sharpe**: Risk-adjusted returns per unit of volatility.
        - **Sortino Ratio**: Risk-adjusted return accounting only for downside volatility.
        - **Treynor Ratio**: Return per unit of systematic (market) risk.
        - **Information Ratio**: Excess return per unit of tracking error.

        - **Sterling Ratio**: Return per unit of average drawdown.
        - **Calmar Ratio**: Return per unit of maximum drawdown.
    """
    # ======= 0. Initialization =======
    frequence_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequence = frequence_dict[frequence]

    cumulative_returns = (1 + returns).cumprod()

    # ======= I. Returns distribution statistics =======
    expected_return = returns.mean() * adjusted_frequence
    volatility = returns.std() * np.sqrt(adjusted_frequence)
    downside_deviation = returns[returns < 0].std() * np.sqrt(adjusted_frequence) if returns[returns < 0].sum() != 0 else 0
    median_return = returns.median() * adjusted_frequence
    skew = returns.skew()
    kurtosis = returns.kurtosis()

    # ======= II. Risk measures =======
    # ------ Maximum Drawdown and Duration
    running_max = cumulative_returns.cummax().replace(0, 1e-10)
    drawdown = (cumulative_returns / running_max) - 1
    drawdown_durations = (drawdown < 0).astype(int).groupby((drawdown == 0).cumsum()).cumsum()

    maximum_drawdown = drawdown.min()
    max_drawdown_duration = drawdown_durations.max()

    # ------ Value at Risk and Conditional Value at Risk
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()

    # ======= III. Market sensitivity measures =======
    if market_returns is None:
        beta = 0
        alpha = 0
        upside_capture = 0
        downside_capture = 0
        tracking_error = 0
    else:
        # ------ Beta and Alpha (Jensens's)
        beta = returns.cov(market_returns) / market_returns.var()
        alpha = expected_return - beta * (market_returns.mean() * adjusted_frequence)

        # ------ Capture Ratios
        upside_capture = returns[market_returns > 0].mean() / market_returns[market_returns > 0].mean()
        downside_capture = returns[market_returns < 0].mean() / market_returns[market_returns < 0].mean()

        # ------ Tracking Error
        tracking_error = returns.sub(market_returns).std() * np.sqrt(adjusted_frequence)

    # ======= IV. Performance measures =======
    # ------ Sharpe, Sortino, Treynor, and Information Ratios
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility != 0 else 0
    sortino_ratio = expected_return / downside_deviation if downside_deviation != 0 else 0
    treynor_ratio = expected_return / beta if beta != 0 else 0
    information_ratio = (expected_return - market_returns.mean() * adjusted_frequence) / tracking_error if tracking_error != 0 else 0

    # ------ Sterling, and Calmar Ratios
    average_drawdown = abs(drawdown[drawdown < 0].mean()) if drawdown[drawdown < 0].sum() != 0 else 0
    sterling_ratio = (expected_return - risk_free_rate) / average_drawdown if average_drawdown != 0 else 0
    calmar_ratio = expected_return / abs(maximum_drawdown) if maximum_drawdown != 0 else 0

    # ======= IV. Store the statistics =======
    stats = {
        "expected_return": expected_return,
        "volatility": volatility,
        "downside_deviation": downside_deviation,
        "median_return": median_return,
        "skew": skew,
        "kurtosis": kurtosis,
        "maximum_drawdown": maximum_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "beta": beta,
        "alpha": alpha,
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
        "tracking_error": tracking_error,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "treynor_ratio": treynor_ratio,
        "information_ratio": information_ratio,
        "sterling_ratio": sterling_ratio,
        "calmar_ratio": calmar_ratio,
    }

    return stats


# ---------------------------------------------------------------------------------------------------------- #
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


# ========================================================
# ==================== MAIN FUNCTIONS ====================
