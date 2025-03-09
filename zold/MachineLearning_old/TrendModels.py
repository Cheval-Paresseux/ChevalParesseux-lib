import os
import sys

sys.path.append(os.path.abspath("../../.."))
import TrainingPreparation.SplitTS as split
import TrainingPreparation.TrainPrepTS as prep
import MachineLearning.ClassifiersTS as classifiers
import Labelling.ApplyLabels as labels
import Features.ApplyFeatures as features
import FinanceTools.Statistics as stats

import pandas as pd
from tqdm import tqdm


class DirectionalModel:
    def __init__(self, data_list: list, params: dict, cluster_index: int):
        # ======= I. RAW SAMPLES =======
        self.params = params
        self.data_list = data_list
        self.cluster_index = cluster_index

        PRIMARY_sample, SECONDARY_sample, VALIDATION_sample, TEST_sample, COVID_sample, PRODUCTION_sample = self.splitData()
        self.PRIMARY_sample = PRIMARY_sample
        self.SECONDARY_sample = SECONDARY_sample
        self.VALIDATION_sample = VALIDATION_sample
        self.TEST_sample = TEST_sample
        self.COVID_sample = COVID_sample
        self.PRODUCTION_sample = PRODUCTION_sample

        self.samples = {"PRIMARY": PRIMARY_sample, "SECONDARY": SECONDARY_sample, "VALIDATION": VALIDATION_sample, "TEST": TEST_sample, "COVID": COVID_sample, "PRODUCTION": PRODUCTION_sample}

        # ======= II. PRIMARY MODELS =======
        self.PRIMARY_trainingDF = None
        self.PRIMARY_training_metrics = None
        self.PRIMARY_classifier = None

        # ======= III. PREDICTIONS =======
        self.SECONDARYsample_PredDirectional = None
        self.VALIDATIONsample_PredDirectional = None
        self.TESTsample_PredDirectional = None
        self.COVIDsample_PredDirectional = None
        self.PRODUCTIONsample_PredDirectional = None

    # -----------------------------------------------------------
    def splitData(self):
        # ======= I. GENERAL SPLIT =======
        general_training, general_validation, general_testing, general_production, general_covid = split.cartesius_split(dfs_list=self.data_list)

        # ======= II. TRAINING SPLIT FOR META-TRAINING =======
        PRIMARY_training, SECONDARY_training, _ = split.proportional_split(training_proportion=0.6, validation_proportion=0.4, dfs_list=general_training)

        # ======= III. CLUSTERING ASSETS FOR STACKING =======
        # III.1 Extract the last date of the general training set
        max_date = pd.Timestamp("1950-01-01")
        for asset in general_training:
            last_date = asset.index[-1]
            if last_date > max_date:
                max_date = last_date

        # III.2 Perform the clustering on the general training set
        price_series = [asset[asset.columns[0]] for asset in general_training]
        data_df = pd.concat(price_series, axis=1).dropna(axis=1)
        max_cluster_length = self.params["max_cluster_length"]

        PRIMARY_sample, clusters_name = prep.clustering_trainingSet(priceSeries_df=data_df, training_dfs_list=PRIMARY_training, last_date=max_date, max_length=max_cluster_length)

        # ======= IV. GROUP THE ASSETS FOR EVERY SAMPLE =======
        def groupAssets(samples_list: list, clusters_name: list):
            samples_clusters_list = []

            # Loop over the clusters
            for asset_list in clusters_name:
                cluster_samples = []

                # Find the assets in the cluster
                for asset in samples_list:
                    if asset.columns[0] in asset_list:
                        cluster_samples.append(asset)

                samples_clusters_list.append(cluster_samples)

            return samples_clusters_list

        SECONDARY_sample = groupAssets(SECONDARY_training, clusters_name)
        VALIDATION_sample = groupAssets(general_validation, clusters_name)
        TEST_sample = groupAssets(general_testing, clusters_name)
        COVID_sample = groupAssets(general_covid, clusters_name)
        PRODUCTION_sample = groupAssets(general_production, clusters_name)

        return PRIMARY_sample, SECONDARY_sample, VALIDATION_sample, TEST_sample, COVID_sample, PRODUCTION_sample

    # -----------------------------------------------------------
    def prep_trainingData(self):
        # ======= I. Extract Parameters =======
        rf_n_components = self.params["rf_n_components"]
        pca_n_components = self.params["pca_n_components"]
        rebalancing_strategy = self.params["rebalancing_strategy"]

        # ======= II. Prepare Training Data =======
        training_data_list = self.PRIMARY_sample[self.cluster_index]

        training_data = prep.vertical_stacking(training_data_dfs_list=training_data_list, rebalancing_strategy=rebalancing_strategy)
        # filtered_training_data = prep.rfImportance_selection(training_data_df=training_data, n_components=rf_n_components)
        # filtered_training_data = prep.pca_selection(training_data_df=filtered_training_data, n_components=pca_n_components)
        filtered_training_data = prep.descent_selection(training_data_df=training_data, n_components=rf_n_components)

        self.PRIMARY_trainingDF = filtered_training_data

        return filtered_training_data

    # -----------------------------------------------------------
    def trainModels(self):
        # ======= I. Extract Parameters =======
        cross_validation_splits = self.params["cross_validation_splits"]
        cross_validation_gap = self.params["cross_validation_gap"]
        max_comb = self.params["max_comb"]
        scoring_function = self.params["scoring_function"]
        n_jobs = self.params["n_jobs"]

        # ======= II. Extract the training DataFrame and the Validation Sample =======
        trainingDF = self.PRIMARY_trainingDF
        validation_sample = self.SECONDARY_sample

        # ======= III. Train the Models =======
        classifier = classifiers.Roseau(training_df=trainingDF, testing_dfs_list=validation_sample, cross_validation_splits=cross_validation_splits, cross_validation_gap=cross_validation_gap)

        _, randomForest_train_metrics = classifier.train_RandomForestClassifier(max_comb=max_comb, scoring_function=scoring_function, n_jobs=n_jobs)
        _, mlp_train_metrics = classifier.train_MLP(max_comb=max_comb, scoring_function=scoring_function, n_jobs=n_jobs)
        _, logisticRegression_train_metrics = classifier.train_LogisticRegression(max_comb=max_comb, scoring_function=scoring_function, n_jobs=n_jobs)
        _, svc_train_metrics = classifier.train_SVC(max_comb=max_comb, scoring_function=scoring_function, n_jobs=n_jobs)

        # ======= IV. Save the Models and Metrics =======
        PRIMARY_training_metrics = {"RandomForest": randomForest_train_metrics, "MLP": mlp_train_metrics, "LogisticRegression": logisticRegression_train_metrics, "SVC": svc_train_metrics}

        self.PRIMARY_classifier = classifier
        self.PRIMARY_training_metrics = PRIMARY_training_metrics

        return classifier, PRIMARY_training_metrics

    # -----------------------------------------------------------
    def makePredictions(self):
        # ======= I. Extract the Classifier =======
        classifier = self.PRIMARY_classifier

        # ======= II. Extract the Testing Data =======
        results = {}
        for sample in self.samples:
            # II.1 Skip the training sample
            if sample == "PRIMARY":
                continue

            # II.2 Extract the testing data
            testing_data_list = self.samples[sample][self.cluster_index]

            # II.3 Make the predictions
            randomForest_results, randomForest_test_metrics = classifier.test_set(model_name="RandomForestClassifier", testing_dfs_list=testing_data_list, meta_label=False)
            mlp_results, mlp_test_metrics = classifier.test_set(model_name="MLP", testing_dfs_list=testing_data_list, meta_label=False)
            logisticRegression_results, logisticRegression_test_metrics = classifier.test_set(model_name="LogisticRegression", testing_dfs_list=testing_data_list, meta_label=False)
            svc_results, svc_test_metrics = classifier.test_set(model_name="SVC", testing_dfs_list=testing_data_list, meta_label=False)

            # II.4 Save the results and metrics
            predictions = {"RandomForest": randomForest_results, "MLP": mlp_results, "LogisticRegression": logisticRegression_results, "SVC": svc_results}
            predictions_metrics = {"RandomForest": randomForest_test_metrics, "MLP": mlp_test_metrics, "LogisticRegression": logisticRegression_test_metrics, "SVC": svc_test_metrics}

            results[sample] = {"predictions": predictions, "metrics": predictions_metrics}

        # ======= III. Save the results =======
        self.SECONDARYsample_PredDirectional = results["SECONDARY"]
        self.VALIDATIONsample_PredDirectional = results["VALIDATION"]
        self.TESTsample_PredDirectional = results["TEST"]
        self.COVIDsample_PredDirectional = results["COVID"]
        self.PRODUCTIONsample_PredDirectional = results["PRODUCTION"]

        return results


class MetaModel:
    def __init__(self, params: dict, SECONDARYsample_modelPredDirectional: list, VALIDATIONsample_modelPredDirectional: list, TESTsample_modelPredDirectional: list, COVIDsample_modelPredDirectional: list, PRODUCTIONsample_modelPredDirectional: list):
        # ======= I. RAW SAMPLES =======
        self.params = params
        self.SECONDARYsample_modelPredDirectional = SECONDARYsample_modelPredDirectional
        self.VALIDATIONsample_modelPredDirectional = VALIDATIONsample_modelPredDirectional
        self.TESTsample_modelPredDirectional = TESTsample_modelPredDirectional
        self.COVIDsample_modelPredDirectional = COVIDsample_modelPredDirectional
        self.PRODUCTIONsample_modelPredDirectional = PRODUCTIONsample_modelPredDirectional

        self.samples = {"SECONDARY": SECONDARYsample_modelPredDirectional, "VALIDATION": VALIDATIONsample_modelPredDirectional, "TEST": TESTsample_modelPredDirectional, "COVID": COVIDsample_modelPredDirectional, "PRODUCTION": PRODUCTIONsample_modelPredDirectional}

        # ======= II. PREPARED META DATA =======
        self.meta_SECONDARYsample = None
        self.meta_VALIDATIONsample = None
        self.meta_TESTsample = None
        self.meta_COVIDsample = None
        self.meta_PRODUCTIONsample = None

        self.meta_samples = None

        # ======= III. META-MODELS =======
        self.meta_trainingDF = None
        self.meta_training_metrics = None
        self.meta_classifier = None

        self.meta_predictions = None

        # ======= IV. PREDICTIONS =======
        self.VALIDATIONsample_PredMeta = None
        self.TESTsample_PredMeta = None
        self.COVIDsample_PredMeta = None
        self.PRODUCTIONsample_PredMeta = None

    # -----------------------------------------------------------
    def prep_metaData(self):
        # ======= I. Extract Parameters =======
        labelling_method = self.params["labelling_method"]
        features_params = self.params["features_params"]

        # ======= II. Prepare the Meta-Data for each Sample =======
        prepared_data = {}
        for sample in self.samples:
            # II.1 Apply meta labelling
            sample_data = self.samples[sample]
            labeled_sample = labels.meta_labelling(individual_dfs_list=sample_data, labelling_method=labelling_method)

            # II.2 Apply meta features
            prepared_sample = []
            for asset in labeled_sample:
                featured_asset = features.predictionsTS_features(predictions_df=asset, features_params=features_params)
                featured_asset = featured_asset.dropna(axis=0)
                prepared_sample.append(featured_asset)

            prepared_data[sample] = prepared_sample

        # ======= III. Save the Meta-Data =======
        self.meta_SECONDARYsample = prepared_data["SECONDARY"]
        self.meta_VALIDATIONsample = prepared_data["VALIDATION"]
        self.meta_TESTsample = prepared_data["TEST"]
        self.meta_COVIDsample = prepared_data["COVID"]
        self.meta_PRODUCTIONsample = prepared_data["PRODUCTION"]

        self.meta_samples = {"SECONDARY": self.meta_SECONDARYsample, "VALIDATION": self.meta_VALIDATIONsample, "TEST": self.meta_TESTsample, "COVID": self.meta_COVIDsample, "PRODUCTION": self.meta_PRODUCTIONsample}

        return prepared_data

    # -----------------------------------------------------------
    def prep_trainingData(self):
        # ======= I. Extract Parameters =======
        rf_n_components = self.params["rf_n_components"]
        pca_n_components = self.params["pca_n_components"]
        rebalancing_strategy = self.params["rebalancing_strategy"]

        # ======= II. Prepare Training Data =======
        training_data_list = self.meta_SECONDARYsample

        training_data = prep.vertical_stacking(training_data_dfs_list=training_data_list, rebalancing_strategy=rebalancing_strategy)
        training_data = training_data.drop(columns=["label"])
        training_data.rename(columns={"meta_label": "label"}, inplace=True)

        # filtered_training_data = prep.rfImportance_selection(training_data_df=training_data, n_components=rf_n_components)
        # filtered_training_data = prep.pca_selection(training_data_df=filtered_training_data, n_components=pca_n_components)
        filtered_training_data = prep.descent_selection(training_data_df=training_data, n_components=rf_n_components)

        self.meta_trainingDF = filtered_training_data

        return filtered_training_data

    # -----------------------------------------------------------
    def trainModels(self):
        # ======= I. Extract Parameters =======
        cross_validation_splits = self.params["cross_validation_splits"]
        cross_validation_gap = self.params["cross_validation_gap"]
        max_comb = self.params["max_comb"]
        scoring_function = self.params["scoring_function"]
        n_jobs = self.params["n_jobs"]

        # ======= II. Extract the training DataFrame and the Validation Sample =======
        trainingDF = self.meta_trainingDF
        validation_sample = self.VALIDATIONsample_modelPredDirectional

        # ======= III. Train the Models =======
        classifier = classifiers.Roseau(training_df=trainingDF, testing_dfs_list=validation_sample, cross_validation_splits=cross_validation_splits, cross_validation_gap=cross_validation_gap)

        _, randomForest_train_metrics = classifier.train_RandomForestClassifier(max_comb=max_comb, scoring_function=scoring_function, n_jobs=n_jobs)
        _, mlp_train_metrics = classifier.train_MLP(max_comb=max_comb, scoring_function=scoring_function, n_jobs=n_jobs)
        _, logisticRegression_train_metrics = classifier.train_LogisticRegression(max_comb=max_comb, scoring_function=scoring_function, n_jobs=n_jobs)
        _, svc_train_metrics = classifier.train_SVC(max_comb=max_comb, scoring_function=scoring_function, n_jobs=n_jobs)

        # ======= IV. Save the Models and Metrics =======
        meta_training_metrics = {"RandomForest": randomForest_train_metrics, "MLP": mlp_train_metrics, "LogisticRegression": logisticRegression_train_metrics, "SVC": svc_train_metrics}

        self.meta_classifier = classifier
        self.meta_training_metrics = meta_training_metrics

        return classifier, meta_training_metrics

    # -----------------------------------------------------------
    def makePredictions(self):
        # ======= I. Extract the Classifier =======
        classifier = self.meta_classifier

        # ======= II. Extract the Testing Data =======
        results = {}
        for sample in self.meta_samples:
            # II.1 Skip the training sample
            if sample == "SECONDARY":
                continue

            # II.2 Extract the testing data
            testing_data_list = self.meta_samples[sample]

            # II.3 Make the predictions
            randomForest_results, randomForest_test_metrics = classifier.test_set(model_name="RandomForestClassifier", testing_dfs_list=testing_data_list, meta_label=True)
            mlp_results, mlp_test_metrics = classifier.test_set(model_name="MLP", testing_dfs_list=testing_data_list, meta_label=True)
            logisticRegression_results, logisticRegression_test_metrics = classifier.test_set(model_name="LogisticRegression", testing_dfs_list=testing_data_list, meta_label=True)
            svc_results, svc_test_metrics = classifier.test_set(model_name="SVC", testing_dfs_list=testing_data_list, meta_label=True)

            # II.4 Save the results and metrics
            predictions = {"RandomForest": randomForest_results, "MLP": mlp_results, "LogisticRegression": logisticRegression_results, "SVC": svc_results}
            predictions_metrics = {"RandomForest": randomForest_test_metrics, "MLP": mlp_test_metrics, "LogisticRegression": logisticRegression_test_metrics, "SVC": svc_test_metrics}

            results[sample] = {"predictions": predictions, "metrics": predictions_metrics}

        # ======= III. Save the results =======
        self.VALIDATIONsample_PredMeta = results["VALIDATION"]
        self.TESTsample_PredMeta = results["TEST"]
        self.COVIDsample_PredMeta = results["COVID"]
        self.PRODUCTIONsample_PredMeta = results["PRODUCTION"]

        return results


class SquadModel:
    def __init__(self, data_list: list, directional_params: dict, meta_params: dict, cluster_index: int):
        # ======= I. INPUTS =======
        self.data_list = data_list
        self.directional_params = directional_params
        self.meta_params = meta_params
        self.cluster_index = cluster_index

        # ======= II. DIRECTIONAL MODEL =======
        self.directionalModel = None
        self.best_directionalModel_name = None

        self.SECONDARYsample_modelPredDirectional = None
        self.VALIDATIONsample_modelPredDirectional = None
        self.TESTsample_modelPredDirectional = None
        self.COVIDsample_modelPredDirectional = None
        self.PRODUCTIONsample_modelPredDirectional = None

        # ======= III. META MODEL =======
        self.metaModel = None
        self.best_metaModel_name = None

        self.VALIDATIONsample_modelPredMeta = None
        self.TESTsample_modelPredMeta = None
        self.COVIDsample_modelPredMeta = None
        self.PRODUCTIONsample_modelPredMeta = None

    # -----------------------------------------------------------
    def directionalWork(self, metric: str):
        # ======= I. DIRECTIONAL MODEL : Make Predictions =======
        directionalModel = DirectionalModel(data_list=self.data_list, params=self.directional_params, cluster_index=self.cluster_index)
        _ = directionalModel.prep_trainingData()
        PRIMARY_classifier, PRIMARY_training_metrics = directionalModel.trainModels()
        directional_results = directionalModel.makePredictions()

        # ======= II. SELECT BEST ML CLASSIFIER =======
        available_models = ["RandomForest", "MLP", "LogisticRegression", "SVC"]
        best_model_name = None
        best_model_metric = -100
        best_results = None

        for model_name in available_models:
            # II.1 Extract the model results
            samples_directional_results = {}
            for sample in directional_results:
                model_results = directional_results[sample]["predictions"][model_name]
                samples_directional_results[sample] = model_results

            SECONDARYsample_modelPredDirectional = [samples_directional_results["SECONDARY"][asset] for asset in samples_directional_results["SECONDARY"]]

            # II.2 Compute the Sharpe Ratio on the validation set
            nb_assets = len(SECONDARYsample_modelPredDirectional)
            assets_results = {}

            # II.2.i Extract the predicted returns for each asset
            for i in range(nb_assets):
                results_df = pd.DataFrame()
                asset_name = SECONDARYsample_modelPredDirectional[i].columns[0]

                results_df[asset_name] = SECONDARYsample_modelPredDirectional[i][asset_name]
                results_df["label"] = SECONDARYsample_modelPredDirectional[i]["label"]
                results_df["directional"] = SECONDARYsample_modelPredDirectional[i]["predictions"]

                results_df["asset_returns"] = results_df[asset_name].pct_change()
                results_df["directional_returns"] = results_df["asset_returns"].shift(-1) * results_df["directional"]

                assets_results[asset_name] = results_df

            # II.2.ii Compute the cluster statistics
            computed_metric = self.get_metrics(assets_results, metric, "directional")

            # II.3 Save the best model
            if computed_metric > best_model_metric:
                best_model_metric = computed_metric
                best_model_name = model_name
                best_results = assets_results

                SECONDARYsample_modelPredDirectional = [samples_directional_results["SECONDARY"][asset] for asset in samples_directional_results["SECONDARY"]]
                VALIDATIONsample_modelPredDirectional = [samples_directional_results["VALIDATION"][asset] for asset in samples_directional_results["VALIDATION"]]
                TESTsample_modelPredDirectional = [samples_directional_results["TEST"][asset] for asset in samples_directional_results["TEST"]]
                COVIDsample_modelPredDirectional = [samples_directional_results["COVID"][asset] for asset in samples_directional_results["COVID"]]
                PRODUCTIONsample_modelPredDirectional = [samples_directional_results["PRODUCTION"][asset] for asset in samples_directional_results["PRODUCTION"]]

        # ======= III. KEEP THE PREDICTIONS FROM THE BEST MODEL =======
        self.directionalModel = directionalModel
        self.best_directionalModel_name = best_model_name
        self.SECONDARYsample_modelPredDirectional = SECONDARYsample_modelPredDirectional
        self.VALIDATIONsample_modelPredDirectional = VALIDATIONsample_modelPredDirectional
        self.TESTsample_modelPredDirectional = TESTsample_modelPredDirectional
        self.COVIDsample_modelPredDirectional = COVIDsample_modelPredDirectional
        self.PRODUCTIONsample_modelPredDirectional = PRODUCTIONsample_modelPredDirectional

        return best_results, best_model_name, best_model_metric, PRIMARY_classifier, PRIMARY_training_metrics

    # -----------------------------------------------------------
    def metaWork(self, metric: str):
        available_labellers = ["right_wrong", "right_wrong_noZero", "trade_lock", "good_bad_ugly", "good_bad_ugly_noZero", "gbu_extended", "gbu_extended_noZero"]
        best_model = None
        best_model_name = None
        best_labeller = None
        best_model_metric = -100
        best_results = None

        for labeller in tqdm(available_labellers):
            self.meta_params["labelling_method"] = labeller
            # ======= I. META MODEL : Make Predictions =======
            metaModel = MetaModel(
                params=self.meta_params,
                SECONDARYsample_modelPredDirectional=self.SECONDARYsample_modelPredDirectional,
                VALIDATIONsample_modelPredDirectional=self.VALIDATIONsample_modelPredDirectional,
                TESTsample_modelPredDirectional=self.TESTsample_modelPredDirectional,
                COVIDsample_modelPredDirectional=self.COVIDsample_modelPredDirectional,
                PRODUCTIONsample_modelPredDirectional=self.PRODUCTIONsample_modelPredDirectional,
            )
            _ = metaModel.prep_metaData()
            _ = metaModel.prep_trainingData()
            meta_classifier, meta_training_metrics = metaModel.trainModels()
            meta_results = metaModel.makePredictions()

            # ======= II. SELECT BEST ML CLASSIFIER =======
            available_models = ["RandomForest", "MLP", "LogisticRegression", "SVC"]

            for model_name in available_models:
                # II.1 Extract the model results
                samples_meta_results = {}
                for sample in meta_results:
                    model_results = meta_results[sample]["predictions"][model_name]
                    samples_meta_results[sample] = model_results

                VALIDATIONsample_modelPredMeta = [samples_meta_results["VALIDATION"][asset] for asset in samples_meta_results["VALIDATION"]]

                # II.2 Compute the Sharpe Ratio on the validation set
                nb_assets = len(VALIDATIONsample_modelPredMeta)
                assets_results = {}

                # II.2.i Extract the predicted returns for each asset
                for i in range(nb_assets):
                    results_df = pd.DataFrame()

                    asset_name = VALIDATIONsample_modelPredMeta[i].columns[0]
                    results_df[asset_name] = VALIDATIONsample_modelPredMeta[i][asset_name]
                    results_df["label"] = VALIDATIONsample_modelPredMeta[i]["label"]
                    results_df["directional"] = self.VALIDATIONsample_modelPredDirectional[i]["predictions"]

                    results_df["meta_label"] = VALIDATIONsample_modelPredMeta[i]["meta_label"]
                    results_df["meta"] = VALIDATIONsample_modelPredMeta[i]["predictions"]  # .apply(lambda x: x if x > 0 else 0)

                    results_df["asset_returns"] = results_df[asset_name].pct_change()
                    results_df["directional_returns"] = results_df["asset_returns"].shift(-1) * results_df["directional"]
                    results_df["meta_returns"] = results_df["directional_returns"] * results_df["meta"]

                    assets_results[asset_name] = results_df

                # II.2.ii Compute the cluster statistics
                computed_metric = self.get_metrics(assets_results, metric, "meta")

                # II.3 Save the best model
                if computed_metric > best_model_metric:
                    best_model = metaModel
                    best_model_metric = computed_metric
                    best_model_name = model_name
                    best_labeller = labeller
                    best_results = assets_results

                    VALIDATIONsample_modelPredMeta = [samples_meta_results["VALIDATION"][asset] for asset in samples_meta_results["VALIDATION"]]
                    TESTsample_modelPredMeta = [samples_meta_results["TEST"][asset] for asset in samples_meta_results["TEST"]]
                    COVIDsample_modelPredMeta = [samples_meta_results["COVID"][asset] for asset in samples_meta_results["COVID"]]
                    PRODUCTIONsample_modelPredMeta = [samples_meta_results["PRODUCTION"][asset] for asset in samples_meta_results["PRODUCTION"]]

        # ======= III. KEEP THE PREDICTIONS FROM THE BEST MODEL =======
        self.metaModel = best_model
        self.best_metaModel_name = best_model_name
        self.best_metaLabeller = best_labeller
        self.VALIDATIONsample_modelPredMeta = VALIDATIONsample_modelPredMeta
        self.TESTsample_modelPredMeta = TESTsample_modelPredMeta
        self.COVIDsample_modelPredMeta = COVIDsample_modelPredMeta
        self.PRODUCTIONsample_modelPredMeta = PRODUCTIONsample_modelPredMeta

        return best_results, best_model_name, best_labeller, best_model_metric, meta_classifier, meta_training_metrics

    # -----------------------------------------------------------
    def get_metrics(self, assets_results: dict, metric: str, model_class: str):
        # ======= I. FINANCIAL METRICS =======
        if model_class == "directional":
            preds_ret = "directional_returns"
            preds = "directional"
            label = "label"
        elif model_class == "meta":
            preds_ret = "meta_returns"
            preds = "meta"
            label = "meta_label"

        cluster_pred = pd.DataFrame()
        cluster_market = pd.DataFrame()
        for asset in assets_results:
            pred_returns = assets_results[asset][preds_ret]
            market_returns = assets_results[asset]["asset_returns"]

            cluster_pred[asset] = pred_returns
            cluster_market[asset] = market_returns

        cluster_pred["average"] = cluster_pred.mean(axis=1)
        cluster_market["average"] = cluster_market.mean(axis=1)

        pred_statistics = stats.compute_stats(returns=cluster_pred["average"], market_returns=cluster_market["average"])
        sharpe_ratio = pred_statistics["sharpe_ratio"]

        # ======= II. MACHINE LEARNING METRICS =======
        MLmetrics = {"accuracy": [], "f1_score": []}
        for asset in assets_results:
            predictions = assets_results[asset][preds]
            labels = assets_results[asset][label]
            accuracy, _, _, f1_score = classifiers.compute_classifyingWeightedStats(predictions, labels)

            MLmetrics["accuracy"].append(accuracy)
            MLmetrics["f1_score"].append(f1_score)

        average_accuracy = sum(MLmetrics["accuracy"]) / len(MLmetrics["accuracy"])
        average_f1_score = sum(MLmetrics["f1_score"]) / len(MLmetrics["f1_score"])

        # ======= III. COMPUTE THE DESIRED METRIC =======
        if metric == "sharpe":
            computed_metric = sharpe_ratio
        elif metric == "accuracy":
            computed_metric = average_accuracy
        elif metric == "f1_score":
            computed_metric = average_f1_score

        return computed_metric

    # -----------------------------------------------------------
    def samples_results(self, sample_predDirectional, sample_predMeta):
        nb_assets = len(sample_predMeta)
        results = {}
        for i in range(nb_assets):
            results_df = pd.DataFrame()

            asset_name = sample_predMeta[i].columns[0]
            results_df[asset_name] = sample_predMeta[i][asset_name]

            results_df["label"] = sample_predDirectional[i]["label"]
            results_df["directional"] = sample_predDirectional[i]["predictions"]

            results_df["meta_label"] = sample_predMeta[i]["meta_label"]
            results_df["meta"] = sample_predMeta[i]["predictions"]  # .apply(lambda x: x if x > 0 else 0)

            results_df["asset_returns"] = results_df[asset_name].pct_change()
            results_df["directional_returns"] = results_df["asset_returns"].shift(-1) * results_df["directional"]
            results_df["meta_returns"] = results_df["directional_returns"] * results_df["meta"]

            results[asset_name] = results_df

        return results

    # -----------------------------------------------------------
    def get_results(self):
        # ======= I. VALIDATION SAMPLE =======
        validation_results = self.samples_results(self.VALIDATIONsample_modelPredDirectional, self.VALIDATIONsample_modelPredMeta)

        # ======= II. TEST SAMPLE =======
        test_results = self.samples_results(self.TESTsample_modelPredDirectional, self.TESTsample_modelPredMeta)

        # ======= III. COVID SAMPLE =======
        covid_results = self.samples_results(self.COVIDsample_modelPredDirectional, self.COVIDsample_modelPredMeta)

        # ======= IV. PRODUCTION SAMPLE =======
        production_results = self.samples_results(self.PRODUCTIONsample_modelPredDirectional, self.PRODUCTIONsample_modelPredMeta)

        return validation_results, test_results, covid_results, production_results
