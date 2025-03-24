import sys
sys.path.append("../")
from Models import common as aux

import numpy as np



#! ==================================================================================== #
#! ================================ Tree Classifiers ================================= #
import numpy as np
import pandas as pd

class DecisionTree(aux.ML_Model):
    def __init__(
        self, 
        max_depth=None, 
        min_samples_split=2,  
        n_features=None        
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    def best_split(self, X, y, n_features, feature_indices, loss_function):
        best_loss = float("inf")
        best_feature = None
        best_threshold = None

        # If n_features is provided, use a subset of features
        if n_features:
            feature_indices = np.random.choice(X.columns, n_features, replace=False)

        for feature in feature_indices:
            values = X[feature].unique()
            for val in values:
                left_mask = X[feature] <= val
                right_mask = ~left_mask
                
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                
                loss_left = loss_function(y[left_mask])
                loss_right = loss_function(y[right_mask])
                loss_split = (left_mask.sum() * loss_left + right_mask.sum() * loss_right) / len(y)

                if loss_split < best_loss:
                    best_loss = loss_split
                    best_feature = feature
                    best_threshold = val
        
        return best_feature, best_threshold

    # Build tree recursively
    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return np.argmax(np.bincount(y))

        best_feature, best_threshold = self.best_split(X, y, self.n_features, X.columns, self.gini_impurity)
        
        if best_feature is None:
            return np.argmax(np.bincount(y))

        left_mask = X[best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self.build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self.build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    # Fit the model to the data
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    # Predict a single sample recursively
    def predict_one(self, x, tree):
        if isinstance(tree, dict):
            if x[tree["feature"]] <= tree["threshold"]:
                return self.predict_one(x, tree["left"])
            else:
                return self.predict_one(x, tree["right"])
        return tree

    # Predict for a batch of data
    def predict(self, X):
        return np.array([self.predict_one(row, self.tree) for _, row in X.iterrows()])

def test(y):
    pass