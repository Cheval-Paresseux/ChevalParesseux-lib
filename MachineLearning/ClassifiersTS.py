import os 
import pandas as pd
import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


class Roseau:
    def __init__(
        self,
        training_df: pd.DataFrame,
        testing_dfs_list: list,
        cross_validation_splits: int = 3,
        cross_validation_gap: int = 20,
    ):
        # ======= I. Input Data =======
        self.training_df = training_df
        self.testing_dfs_list = testing_dfs_list
        
        # ======= II. Time Series Cross-Validation =======
        self.tscv = TimeSeriesSplit(n_splits=cross_validation_splits, gap=cross_validation_gap)

        # ======= III. Models  =======
        # Non-Linear Models
        self.random_forest = None
        self.mlp = None
        
        # Linear Models
        self.logreg_up = None
        self.logreg_down = None
        self.svc_up = None
        self.svc_down = None

    # ==================== Non-Linear Models ====================
    def train_RandomForestClassifier(
        self,
        max_comb: int,
        scoring_function: str = "f1_macro",
        n_jobs: int = 1,
    ):
        # ======= 0. Initialization =======
        training_data = self.training_df.copy()
        random_forest = RandomForestClassifier(random_state=69)

        # ======= I. Parameters for Randomized Search =======
        unique_classes = set(y.unique())
        if unique_classes == {-1, 0, 1}:  # Trinary classification
            class_weights = ["balanced", {-1: 2, 0: 1, 1: 1}, {-1: 1, 0: 2, 1: 1}, {-1: 1, 0: 1, 1: 2}]
        elif unique_classes == {0, 1}:  # Binary classification
            class_weights = ["balanced", {0: 1, 1: 1}, {0: 2, 1: 1}, {0: 1, 1: 2}]
            
        param_dist = {
            "n_estimators": [100, 150, 200, 300, 500, 1000], 
            "max_depth": [2, 3, 4, 5], 
            "min_samples_split": [10, 15, 20, 25, 30], 
            "min_samples_leaf": [5, 10, 15, 20, 30], 
            "max_features": ["sqrt", "log2"], 
            "class_weight": class_weights
        }

        # ======= II. Train the model =======
        X = training_data.drop(columns=["asset", "label"]).copy()
        y = training_data["label"].copy()
        
        randomized_search = RandomizedSearchCV(
            estimator=random_forest,
            param_distributions=param_dist,
            n_iter=max_comb,  
            cv=self.tscv,  
            scoring=scoring_function,
            verbose=0,
            n_jobs=n_jobs,
            random_state=69,
        )

        randomized_search.fit(X, y)

        # ======= III. Save the best model =======
        model = randomized_search.best_estimator_
        self.random_forest = model
        self.features_used = X.columns
            
        # ======= III. Extract Training Metrics =======
        feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
        training_accuracy = model.score(X, y)
        cv_score = randomized_search.best_score_

        train_metrics = {
            "feature_importances": feature_importances,
            "training_accuracy": training_accuracy,
            "cv_score": cv_score,
            "parameters": randomized_search.best_params_,
        }

        return model, train_metrics
    
    # -----------------------------------------------------------
    def train_MLP(
        self,
        max_comb: int,
        scoring_function: str = "f1_macro",
        n_jobs: int = 1,
    ):
        # ======= 0. Initialization =======
        training_data = self.training_df.copy()
        mlp = MLPClassifier(random_state=69, early_stopping=True, n_iter_no_change=10)

        # ======= I. Initialize Model and Parameters =======
        param_dist = {
            "hidden_layer_sizes": [(50,), (100,), (150,), (200,), (100, 50)],
            "activation": ["relu", "tanh", "logistic"],
            "solver": ["adam", "sgd"],
            "alpha": loguniform(1e-4, 1e-1),
            "learning_rate_init": loguniform(1e-4, 1e-2),
            "max_iter": [100, 200, 300, 500],
        }

        # ======= II. Randomized Search for MLPClassifier =======
        X = training_data.drop(columns=["asset", "label"])
        y = training_data["label"]
        
        randomized_search = RandomizedSearchCV(
            estimator=mlp,
            param_distributions=param_dist,
            n_iter=max_comb,  
            cv=self.tscv,  
            scoring=scoring_function,
            verbose=0,
            random_state=69,
            n_jobs=n_jobs,
        )
        
        randomized_search.fit(X, y)

        # ======= III. Save the best model =======
        model = randomized_search.best_estimator_
        self.mlp = model
        self.features_used = X.columns
            
        # ======= III. Extract Training Metrics =======
        training_accuracy = model.score(X, y)
        cv_score = randomized_search.best_score_

        train_metrics = {
            "training_accuracy": training_accuracy,
            "cv_score": cv_score,
            "parameters": randomized_search.best_params_,
        }

        return model, train_metrics

    # ==================== Linear Models ====================
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

