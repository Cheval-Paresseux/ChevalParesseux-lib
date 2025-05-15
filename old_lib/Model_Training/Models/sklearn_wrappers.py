from ..Models import common as com
from ...Data_Processing.Measures import Entropy as ent

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#! ==================================================================================== #
#! ================================ Tree Classifiers ================================== #
class skLearnTreeClassifier(com.ML_Model):
    """
    Wrapper for sklearn DecisionTreeClassifier.
    
    This class is used to create a decision tree classifier using the sklearn library, it is fitted to be integrated to the ML_Model framework.
    """
    def __init__(
        self, 
        n_jobs: int = 1,
        random_state: int = 72
    ):
        """
        Parameters:
            - n_jobs (int): Not useful for this model, but kept for the sake of consistency with the other models.
            - random_state (int): Random state for reproducibility.
        """  
        # ======= I. Hyper Parameters ======= 
        self.params = None
        self.n_jobs = n_jobs
        np.random.seed(random_state)
        
        # ======= II. Variables ======= 
        self.decision_tree = None
        self.min_proba = None
        self.raw_predict = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        raw_predict: bool = False,
        min_proba: float = 0.5,
        criterion: str = 'gini',
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int = None,
        ccp_alpha: float = 0.0,
        class_weight: str = None,
    ):
        """
        Set the hyperparameters for the DecisionTreeClassifier.
        
        Parameters:
            - criterion (str): The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
            - max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            - min_samples_split (int): The minimum number of samples required to split an internal node.
            - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            - max_features (int): The number of features to consider when looking for the best split.
        """
        self.params = {
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'ccp_alpha': ccp_alpha,
            'class_weight': class_weight
        }
        
        self.min_proba = min_proba
        self.raw_predict = raw_predict
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(self, X_train, y_train):
        """
        Process the data to be used for training the model.
        
        Parameters:
            - X_train (pd.DataFrame): The training data features.
            - y_train (pd.Series): The training data labels.
        """
        X, y = np.array(X_train), np.array(y_train)
        
        return X, y
    
    #?____________________________________________________________________________________ #
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame): The training data features.
            - y_train (pd.Series): The training data labels.
        """
        # ======= I. Data Processing ======= 
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Model Fitting =======
        clf = tree.DecisionTreeClassifier(**self.params)
        clf = clf.fit(X, y)
        
        # ======= III. Model Saving =======
        self.decision_tree = clf
        self.feature_importances = clf.feature_importances_
        
        return self.feature_importances
    
    #?____________________________________________________________________________________ #
    def predict(self, X: pd.DataFrame):
        """
        Predict the labels for the given data.
        
        Parameters:
            - X (pd.DataFrame): The data to be predicted.
        """
        X_test = np.array(X)
        preds_probas = self.decision_tree.predict_proba(X_test)
        
        # Align predicted indices with actual class labels
        class_order = self.decision_tree.classes_
        predicted_class_indices = np.argmax(preds_probas, axis=1)
        predicted_labels = class_order[predicted_class_indices]

        # Apply min_proba threshold
        max_probs = np.max(preds_probas, axis=1)
        filtered_preds = np.where(max_probs >= self.min_proba, predicted_labels, np.nan)

        # Final predictions, filled with 0 where confidence is too low
        predictions = pd.Series(filtered_preds, index=X.index).fillna(0)
        
        return predictions

#*____________________________________________________________________________________ #
class skLearnRFClassifier(com.ML_Model):
    """
    Wrapper for sklearn RandomForestClassifier.

    This class creates a random forest classifier using sklearn, designed to integrate with the ML_Model framework.
    """
    def __init__(
        self, 
        n_jobs: int = 1,
        random_state: int = 72
    ):
        """
        Parameters:
            - n_jobs (int): Number of parallel jobs.
            - random_state (int): Random seed for reproducibility.
        """
        self.params = None
        self.n_jobs = n_jobs
        np.random.seed(random_state)
        
        self.random_forest = None

    #?____________________________________________________________________________________ #
    def set_params(
        self,
        n_estimators: int = 100,
        criterion: str = 'gini',
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        class_weight: str = None,
        bootstrap: bool = True,
        max_samples: float = None,
    ):
        """
        Set hyperparameters for the RandomForestClassifier.

        Parameters mirror sklearn's RandomForestClassifier.
        """
        self.params = {
            'n_estimators': n_estimators,
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight,
            'bootstrap': bootstrap,
            'max_samples': max_samples,
            'n_jobs': self.n_jobs,
            'random_state': 72
        }
        
        return self

    #?____________________________________________________________________________________ #
    def process_data(self, X_train, y_train):
        X, y = np.array(X_train), np.array(y_train)
        return X, y

    #?____________________________________________________________________________________ #
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X, y = self.process_data(X_train, y_train)
        
        clf = RandomForestClassifier(**self.params)
        clf = clf.fit(X, y)
        
        self.random_forest = clf
        self.feature_importances = clf.feature_importances_
        
        return self.feature_importances

    #?____________________________________________________________________________________ #
    def predict(self, X: pd.DataFrame):
        X_test = np.array(X)
        
        predictions = self.random_forest.predict(X_test)
        predictions = pd.Series(predictions, index=X.index)

        return predictions
    
#*____________________________________________________________________________________ #
class skLearnLogRegClassifier(com.ML_Model):
    """
    Wrapper for sklearn LogisticRegression.

    This class creates a logistic regression classifier using sklearn, fitted to integrate into the ML_Model framework.
    """
    def __init__(
        self,
        n_jobs: int = 1,
        random_state: int = 72
    ):
        self.params = None
        self.n_jobs = n_jobs
        np.random.seed(random_state)
        
        self.log_reg = None

    #?____________________________________________________________________________________ #
    def set_params(
        self,
        penalty: str = 'l2',
        C: float = 1.0,
        fit_intercept: bool = True,
        solver: str = 'lbfgs',
        max_iter: int = 100,
        class_weight: str = None,
    ):
        """
        Set hyperparameters for the LogisticRegression model.
        """
        self.params = {
            'penalty': penalty,
            'C': C,
            'fit_intercept': fit_intercept,
            'solver': solver,
            'max_iter': max_iter,
            'class_weight': class_weight,
            'n_jobs': self.n_jobs,
            'random_state': 72
        }

        return self

    #?____________________________________________________________________________________ #
    def process_data(self, X_train, y_train):
        X, y = np.array(X_train), np.array(y_train)
        return X, y

    #?____________________________________________________________________________________ #
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X, y = self.process_data(X_train, y_train)
        
        clf = LogisticRegression(**self.params)
        clf = clf.fit(X, y)
        
        self.log_reg = clf
        if hasattr(clf, 'coef_'):
            self.feature_importances = np.abs(clf.coef_).flatten()
        else:
            self.feature_importances = None

        return self.feature_importances

    #?____________________________________________________________________________________ #
    def predict(self, X: pd.DataFrame):
        X_test = np.array(X)
        
        predictions = self.log_reg.predict(X_test)
        predictions = pd.Series(predictions, index=X.index)

        return predictions
    
    