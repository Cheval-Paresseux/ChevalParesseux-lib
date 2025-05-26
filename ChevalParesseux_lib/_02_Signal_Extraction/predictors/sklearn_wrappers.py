from ..predictors import common as com

import numpy as np
import pandas as pd
from typing import Union, Self
 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



#! ==================================================================================== #
#! ================================ Tree Classifiers ================================== #
class SKL_tree_classifier(com.Model):
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
        Constructor for the SKL_tree_classifier class.
        
        Parameters:
            - n_jobs (int): Not useful for this model, but kept for the sake of consistency with the other models.
            - random_state (int): Random state for reproducibility.
        """  
        # ======= I. Initialization ======= 
        super().__init__(n_jobs=n_jobs)
        np.random.seed(random_state)
        
        self.params = {}
        
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
            - raw_predict (bool): If True, the model will return the raw predictions (probabilities) instead of the predicted classes.
            - min_proba (float): The minimum probability threshold for a prediction to be considered valid. If the maximum probability is below this threshold, the prediction will be set to NaN.
            - criterion (str): The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
            - max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            - min_samples_split (int): The minimum number of samples required to split an internal node.
            - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            - max_features (int): The number of features to consider when looking for the best split.
            - ccp_alpha (float): Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen.
            - class_weight (str): Weights associated with classes. If not given, all classes are supposed to have weight one.
        
        Returns:
            - self: The instance of the class with updated parameters.
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
        
        self.raw_predict = raw_predict
        self.min_proba = min_proba
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        features_matrix: Union[pd.DataFrame, np.ndarray], 
        target_vector: Union[pd.Series, np.ndarray]
    ) -> tuple:
        """
        Process the data to be used for training the model.
        
        Parameters:
            - X_train (pd.DataFrame): The training data features.
            - y_train (pd.Series): The training data labels.
        
        Returns:
            - tuple: A tuple containing the processed features and labels.
        """
        X, y = np.array(features_matrix), np.array(target_vector)
        
        return X, y
    
    #?____________________________________________________________________________________ #
    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray], 
        y_train: Union[pd.Series, np.ndarray]
    ) -> Self:
        """
        Fit the model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame): The training data features.
            - y_train (pd.Series): The training data labels.
        
        Returns:
            - self: The instance of the class with the fitted model.
        """
        # ======= I. Data Processing ======= 
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Model Fitting =======
        clf = tree.DecisionTreeClassifier(**self.params)
        clf = clf.fit(X, y)
        
        # ======= III. Model Saving =======
        self.decision_tree = clf
        self.feature_importances = clf.feature_importances_
        
        return self
    
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: pd.DataFrame
    ) -> pd.Series:
        """
        Predict the labels for the given data.
        
        Parameters:
            - X_test (pd.DataFrame): The data to be predicted.
        
        Returns:
            - pd.Series: The predicted labels.
        """
        # ======= I. Data Processing =======
        X = np.array(X_test)
        
        # ======= II. Predictions =======
        preds_probas = self.decision_tree.predict_proba(X)
        
        # ======= III. Raw Predictions =======
        class_order = self.decision_tree.classes_
        predicted_class_indices = np.argmax(preds_probas, axis=1)
        predicted_labels = class_order[predicted_class_indices]

        # ====== IV. Filter Predictions =======
        if self.raw_predict:
            return predicted_labels
        
        max_probs = np.max(preds_probas, axis=1)
        filtered_preds = np.where(max_probs >= self.min_proba, predicted_labels, np.nan)

        predictions = pd.Series(filtered_preds, index=X_test.index).fillna(0)
        
        return predictions

#*____________________________________________________________________________________ #
class SKL_randomForest_classifier(com.Model):
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
        Constructor for the SKL_randomForest_classifier class.
        
        Parameters:
            - n_jobs (int): Not useful for this model, but kept for the sake of consistency with the other models.
            - random_state (int): Random state for reproducibility.
        """  
        # ======= I. Initialization ======= 
        super().__init__(n_jobs=n_jobs)
        
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.params = {}
        
        # ======= II. Variables ======= 
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

        Parameters :
            - n_estimators (int): The number of trees in the forest.
            - criterion (str): The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
            - max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            - min_samples_split (int): The minimum number of samples required to split an internal node.
            - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            - max_features (str): The number of features to consider when looking for the best split. Can be "sqrt", "log2", or a float representing a percentage.
            - class_weight (str): Weights associated with classes. If not given, all classes are supposed to have weight one.
            - bootstrap (bool): Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
            - max_samples (float): If bootstrap is True, the number of samples to draw to train each base estimator. If None, then draw X.shape[0] samples.
        
        Returns:
            - self: The instance of the class with updated parameters.
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
            'random_state': self.random_state
        }
        
        return self

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        features_matrix: Union[pd.DataFrame, np.ndarray], 
        target_vector: Union[pd.Series, np.ndarray]
    ) -> tuple:
        """
        Process the data to be used for training the model.
        
        Parameters:
            - X_train (pd.DataFrame): The training data features.
            - y_train (pd.Series): The training data labels.
        
        Returns:
            - tuple: A tuple containing the processed features and labels.
        """
        X, y = np.array(features_matrix), np.array(target_vector)
        
        return X, y

    #?____________________________________________________________________________________ #
    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray], 
        y_train: Union[pd.Series, np.ndarray]
    ) -> Self:
        """
        Fit the model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame): The training data features.
            - y_train (pd.Series): The training data labels.
        
        Returns:
            - self: The instance of the class with the fitted model.
        """
        # ======= I. Data Processing ======= 
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Model Fitting =======
        clf = RandomForestClassifier(**self.params)
        clf = clf.fit(X, y)
        
        # ======= III. Model Saving =======
        self.random_forest = clf
        self.feature_importances = clf.feature_importances_
        
        return self

    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: pd.DataFrame
    ) -> pd.Series:
        """
        Predict the labels for the given data.
        
        Parameters:
            - X_test (pd.DataFrame): The data to be predicted.
        
        Returns:
            - pd.Series: The predicted labels.
        """
        # ======= I. Data Processing =======
        X = np.array(X_test)
        
        # ======= II. Predictions =======
        predictions = self.random_forest.predict(X)
        predictions = pd.Series(predictions, index=X_test.index)

        return predictions
    
#*____________________________________________________________________________________ #
class SKL_logisticRegression_classifier(com.Model):
    """
    Wrapper for sklearn LogisticRegression.

    This class creates a logistic regression classifier using sklearn, fitted to integrate into the ML_Model framework.
    """
    def __init__(
        self, 
        n_jobs: int = 1,
        random_state: int = 72
    ):
        """
        Constructor for the SKL_logisticRegression_classifier class.
        
        Parameters:
            - n_jobs (int): Not useful for this model, but kept for the sake of consistency with the other models.
            - random_state (int): Random state for reproducibility.
        """  
        # ======= I. Initialization ======= 
        super().__init__(n_jobs=n_jobs)
        
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.params = {}
        
        # ======= II. Variables ======= 
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
        
        Parameters:
            - penalty (str): Used to specify the norm used in the penalization. The 'newton-cg', 'sag' and 'lbfgs' solvers support only l2 penalties.
            - C (float): Inverse of regularization strength; smaller values specify stronger regularization.
            - fit_intercept (bool): Whether to include an intercept in the model.
            - solver (str): Algorithm to use in the optimization problem. Default is 'lbfgs'.
            - max_iter (int): Maximum number of iterations taken for the solvers to converge.
            - class_weight (str): Weights associated with classes. If not given, all classes are supposed to have weight one.
        
        Returns:
            - self: The instance of the class with updated parameters.
        """
        self.params = {
            'penalty': penalty,
            'C': C,
            'fit_intercept': fit_intercept,
            'solver': solver,
            'max_iter': max_iter,
            'class_weight': class_weight,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state
        }

        return self

    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        features_matrix: Union[pd.DataFrame, np.ndarray], 
        target_vector: Union[pd.Series, np.ndarray]
    ) -> tuple:
        """
        Process the data to be used for training the model.
        
        Parameters:
            - X_train (pd.DataFrame): The training data features.
            - y_train (pd.Series): The training data labels.
        
        Returns:
            - tuple: A tuple containing the processed features and labels.
        """
        X, y = np.array(features_matrix), np.array(target_vector)
        
        return X, y

    #?____________________________________________________________________________________ #
    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray], 
        y_train: Union[pd.Series, np.ndarray]
    ) -> Self:
        """
        Fit the model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame): The training data features.
            - y_train (pd.Series): The training data labels.
        
        Returns:
            - self: The instance of the class with the fitted model.
        """
        # ======= I. Data Processing ======= 
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Model Fitting =======
        clf = LogisticRegression(**self.params)
        clf = clf.fit(X, y)
        
        # ======= III. Model Saving =======
        self.log_reg = clf
        if hasattr(clf, 'coef_'):
            self.feature_importances = np.abs(clf.coef_).flatten()
        else:
            self.feature_importances = None

        return self

    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: pd.DataFrame
    ) -> pd.Series:
        """
        Predict the labels for the given data.
        
        Parameters:
            - X_test (pd.DataFrame): The data to be predicted.
        
        Returns:
            - pd.Series: The predicted labels.
        """
        # ======= I. Data Processing =======
        X = np.array(X_test)
        
        # ======= II. Predictions =======
        predictions = self.log_reg.predict(X)
        predictions = pd.Series(predictions, index=X_test.index)

        return predictions
    
