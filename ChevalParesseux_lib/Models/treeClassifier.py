from ..Models import common as com
from ..Measures import Entropy as ent

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
 

#! ==================================================================================== #
#! ================================ Tree Classifiers ================================== #
class Node:
    """
    This class represents a node in a Decision Tree. It can be a leaf node or a decision node.
    It holds the following attributes:
        - feature: The feature to split on
        - threshold: The threshold to split the feature
        - left: The left child node
        - right: The right child node
        - value: The predicted value if the node is a leaf node
    """
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value=None, samples=None, impurity=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples
        self.impurity = impurity
    
    def is_leaf_node(self):
        return self.value is not None

#*____________________________________________________________________________________ #
class DecisionTreeClassifier(com.ML_Model):
    def __init__(
        self, 
        criterion: str = "gini", 
        max_depth: int = None, 
        min_samples_split: int = 2, 
        min_samples_leaf: int = 1, 
        max_features: int = None,
        n_jobs: int = 1,
        random_state: int = 72
    ):
        # ======= I. Hyper Parameters ======= #
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        np.random.seed(random_state)
        
        # ======= II. Available Entropies ======= #
        self.available_entropies = {
            "gini": ent.get_gini_impurity,
            "entropy": ent.get_shannon_entropy,
        }
        
        # ======= III. Variables ======= #
        self.root = None
        self.feature_importances = None
        
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(self, X_train, y_train):
        X, y = np.array(X_train), np.array(y_train)
        
        return X, y
    
    #?____________________________________________________________________________________ #
    def get_impurity(self, y):
        """
        This function computes the impurity of a node.
            - y (np.array): The target values of the node.
            - self.criterion (str): The criterion to use to compute the impurity.
        """
        if self.criterion in self.available_entropies:
            impurity = self.available_entropies[self.criterion](y)
        else:
            raise ValueError(f"Unknown criterion '{self.criterion}'")

        return impurity
    
    #?____________________________________________________________________________________ #
    def test_split(self, feature, y_sorted, threshold, nb_labels, parent_impurity):
        # ======= I. Initialize the variables ======= #
        left_mask = feature <= threshold
        right_mask = ~left_mask
        
        # ======= II. Check if the split is valid ======= #
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return (-1, threshold)
        
        # ======= III. Compute the impurities and the information gain ======= #
        left_impurity = self.get_impurity(y_sorted[left_mask])
        right_impurity = self.get_impurity(y_sorted[right_mask])
        
        # ======= IV. Compute the information gain ======= #
        child_impurity = (np.sum(left_mask) / nb_labels) * left_impurity + (np.sum(right_mask) / nb_labels) * right_impurity
        information_gain = parent_impurity - child_impurity
        
        return (information_gain, threshold)
    
    #?____________________________________________________________________________________ #
    def get_best_split(self, X, y, features_indexes):
        # ======= I. Initialize the variables ======= #
        best_gain = -1
        split_feature, split_threshold = None, None
        parent_impurity = self.get_impurity(y)
        nb_labels = len(y)

        # ======= II. Precompute sorted features ======= #
        sorted_features = {}
        for feature_idx in features_indexes:
            feature = X[:, feature_idx]
            sorted_idx = np.argsort(feature)
            sorted_features[feature_idx] = (feature[sorted_idx], y[sorted_idx])
        
        # ======= III. Define the Process Feature funciton ======= #
        def process_feature(feature_idx):
            # Get the sorted feature and target values
            feature, y_sorted = sorted_features[feature_idx]
            unique_values = np.unique(feature)
            
            # Check if there is no variance
            if len(unique_values) == 1:
                return None 

            # Compute the possible splits and their information gain
            possible_thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            thresholds_results = [self.test_split(feature, y_sorted, threshold, nb_labels, parent_impurity) for threshold in possible_thresholds]
            
            return feature_idx, thresholds_results
        
        # ======= IV. Process the features in parallel ======= #
        feature_results = Parallel(n_jobs=self.n_jobs)(delayed(process_feature)(feature_idx) for feature_idx in features_indexes)

        # ======= V. Check each result for best gain ======= #
        for feature_idx, thresholds_results in feature_results:
            if thresholds_results is None:
                continue  # Skip if no valid results
            
            for information_gain, threshold in thresholds_results:
                if information_gain > best_gain:
                    best_gain = information_gain
                    split_feature, split_threshold = feature_idx, threshold

        return split_feature, split_threshold, best_gain
    
    #?____________________________________________________________________________________ #
    def build_tree(self, X, y, depth=0):
        # ======= I. Initialize the variables ======= #
        nb_samples, nb_features = X.shape
        num_labels = len(np.unique(y))
        impurity = self.get_impurity(y)

        # ======= II. Check Stopping Criteria ======= #
        if (depth >= self.max_depth or num_labels == 1 or nb_samples < self.min_samples_split):
            leaf_value = pd.Series(y).value_counts().idxmax()
            leaf_samples = pd.Series(y).groupby(y).count().tolist()
            node = Node(value=leaf_value, samples=leaf_samples, impurity=impurity)
            return node

        # ======= III. Get a random subset of the features ======= #
        max_features = min(nb_features, self.max_features) if self.max_features else nb_features
        features_subset_indexes = np.random.choice(nb_features, max_features, replace=False)

        # ======= IV. Get the best split ======= #
        best_feature, best_threshold, best_gain = self.get_best_split(X, y, features_subset_indexes)
        
        # ======= V. Check if the current split can't be improved ======= #
        if best_gain == -1:
            # If no good split, return leaf node
            leaf_value = pd.Series(y).value_counts().idxmax()
            leaf_samples = pd.Series(y).groupby(y).count().tolist()
            node = Node(value=leaf_value, samples=leaf_samples, impurity=impurity)
            return node

        # ======= VI. Split the data and build the subtrees ======= #
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # ======= VII. Return the decision node ======= #
        node = Node(best_feature, best_threshold, left_subtree, right_subtree, samples=[np.sum(left_mask), np.sum(right_mask)], impurity=impurity)

        return node

    #?____________________________________________________________________________________ #
    def traverse_tree(self, row, node):
        if node.is_leaf_node():
            # We found a leaf node
            return node.value
        
        elif row[node.feature] <= node.threshold:
            # We go left
            return self.traverse_tree(row, node.left)
        
        else:
            # We go right
            return self.traverse_tree(row, node.right)
    
    #?____________________________________________________________________________________ #
    def get_features_importances(self, X, y):
        # ======= 0. Define the recursive function ======= #
        def compute_importance(node, total_samples):
            # Base case: we reached a leaf node
            if node is None or node.is_leaf_node():
                return

            # Update the importance of the feature used to split the node
            left_samples = np.sum(X[:, node.feature] <= node.threshold)
            right_samples = total_samples - left_samples

            self.features_importances[node.feature] += left_samples + right_samples
            compute_importance(node.left, left_samples)
            compute_importance(node.right, right_samples)

        # ======= I. Initialize the feature importances ======= #
        self.features_importances = np.zeros(X.shape[1])
        
        # ======= II. Compute the feature importances ======= #
        compute_importance(self.root, len(y))
        
        # ======= III. Normalize the feature importances ======= #
        self.features_importances /= np.sum(self.features_importances)
        
        return self.features_importances
    
    #?_____________________________ User Functions _______________________________________ #
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        # ======= I. Process the data ======= #
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Build the tree ======= #
        self.root = self.build_tree(X, y)
        
        # ======= III. Compute the key model statistics ======= #
        features_importances = self.get_features_importances(X, y)
        
        return features_importances

    #?____________________________________________________________________________________ #
    def predict(self, X: pd.DataFrame):
        predictions = np.array([self.traverse_tree(row, self.root) for row in np.array(X)])
        return predictions




