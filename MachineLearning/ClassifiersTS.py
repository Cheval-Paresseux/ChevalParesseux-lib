import os 
import pandas as pd
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
        self.features_used = None
        
        # Non-Linear Models
        self.random_forest = None
        self.mlp = None
        
        # Linear Models
        self.logistic_regression = None
        self.svc = None

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
        
        X = training_data.drop(columns=["asset", "label"]).copy()
        y = training_data["label"].copy()

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
        
        X = training_data.drop(columns=["asset", "label"])
        y = training_data["label"]

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
        max_comb: int,
        scoring_function: str = "f1_macro",
        n_jobs: int = 1,
    ):
        # ======= 0. Initialization =======
        training_data = self.training_df.copy()
        logistic_regression= LogisticRegression(random_state=69)
        
        X = training_data.drop(columns=["asset", "label"]).copy()
        y = training_data["label"].copy()

        # ======= I. Parameters for Randomized Search =======
        unique_classes = set(y.unique())
        if unique_classes == {-1, 0, 1}:  # Trinary classification
            class_weights = ["balanced", {-1: 2, 0: 1, 1: 1}, {-1: 1, 0: 2, 1: 1}, {-1: 1, 0: 1, 1: 2}]
        elif unique_classes == {0, 1}:  # Binary classification
            class_weights = ["balanced", {0: 1, 1: 1}, {0: 2, 1: 1}, {0: 1, 1: 2}]
            
        param_dist = {
            "C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000],
            "max_iter": [100, 200, 500, 1000, 2000, 5000],
            "class_weight": class_weights,
            "solver": ["lbfgs", "newton-cg", "saga"],  
            "tol": [1e-4, 1e-3, 1e-2], 
            "multi_class": ["ovr", "multinomial"],
        }

        # ======= II. Train the model =======
        randomized_search = RandomizedSearchCV(
            estimator=logistic_regression,
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
        self.logistic_regression = model
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

    # -----------------------------------------------------------
    def train_SVC(
        self,
        max_comb: int,
        scoring_function: str = "f1_macro",
        n_jobs: int = 1,
    ):
        # ======= 0. Initialization =======
        training_data = self.training_df.copy()
        svc = SVC(random_state=69, probability=True)
        
        X = training_data.drop(columns=["asset", "label"]).copy()
        y = training_data["label"].copy()

        # ======= I. Parameters for Randomized Search =======
        unique_classes = set(y.unique())
        if unique_classes == {-1, 0, 1}:  # Trinary classification
            class_weights = ["balanced", {-1: 2, 0: 1, 1: 1}, {-1: 1, 0: 2, 1: 1}, {-1: 1, 0: 1, 1: 2}]
        elif unique_classes == {0, 1}:  # Binary classification
            class_weights = ["balanced", {0: 1, 1: 1}, {0: 2, 1: 1}, {0: 1, 1: 2}]
            
        param_dist = {
            "C": [0.001, 0.01, 0.1, 1, 2.5, 5, 10, 100, 1000],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1, 10],  # For rbf & poly
            "degree": [2, 3, 4, 5],  # Only relevant for poly kernel
            "class_weight": class_weights,
        }

        # ======= II. Randomized Search for Support Vector Classifiers =======
        randomized_search = RandomizedSearchCV(
            estimator=svc,
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
        self.svc = model
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

    
    # ===========================================================
    # ==================== TESTING LOGIC ====================
    def test_asset(self, model_name: str, testing_data_df: pd.DataFrame):
        
        # ======= 0. Auxiliary Functions =======
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

        # ======= I. Prepare the Testing Data =======
        asset_name = testing_data_df.columns[0]
        
        kept_columns = [asset_name, "label"] + self.features_used.tolist()
        test_df = testing_data_df[kept_columns].copy()

        
        X = test_df.drop(columns=[asset_name, "label"])
        y = test_df["label"]

        # ======= II. Make Predictions =======
        if model_name == "RandomForestClassifier":
            predictions = self.random_forest.predict(X)

        elif model_name == "LogisticRegression":
            predictions = self.logistic_regression.predict(X)

        elif model_name == "SVC":
            predictions = self.svc.predict(X)

        elif model_name == "MLP":
            predictions = self.mlp.predict(X)

        else:
            raise ValueError("Model name not recognized.")
        
        results_df = pd.DataFrame()
        results_df[asset_name] = testing_data_df[asset_name]
    
        results_df["predictions"] = predictions  # apply_confirmation(predictions)
        results_df["label"] = y

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
    
    def test_set(self, model_name: str):
        testing_results = {}
        testing_metrics = {}

        for testing_data_df in self.testing_dfs_list:
            asset_name = testing_data_df.columns[0]
            
            results_df, test_metrics = self.test_asset(model_name, testing_data_df)
            testing_results[asset_name] = results_df
            testing_metrics[asset_name] = test_metrics

        return testing_results, testing_metrics

    