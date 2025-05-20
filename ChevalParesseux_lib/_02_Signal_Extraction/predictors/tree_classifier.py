from ..predictors import common as com
from ... import utils

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Union, Self



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
    def __init__(
        self, 
        feature = None, 
        threshold = None, 
        left = None, 
        right = None, 
        *, 
        value=None, 
        samples=None, 
        impurity=None
    ) -> None:
        """
        Constructor for the Node class.
        
        Parameters:
            - feature: The feature to split on
            - threshold: The threshold to split the feature
            - left: The left child node
            - right: The right child node
            - value: The predicted value if the node is a leaf node
            - samples: The number of samples in the node
            - impurity: The impurity of the node
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples
        self.impurity = impurity
        
    #?____________________________________________________________________________________ #
    def is_leaf_node(
        self
    ) -> bool:
        return self.value is not None

#*____________________________________________________________________________________ #
class Tree_classifier(com.Model):
    """
    This class implements a Decision Tree Classifier. 
    """
    def __init__(
        self, 
        n_jobs: int = 1,
        random_state: int = 72
    ) -> None:
        """
        Constructor for the Tree_classifier class.
        
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
        self.available_entropies = {
            "gini": utils.get_gini_impurity,
            "shannon": utils.get_shannon_entropy,
            "plugin": utils.get_plugin_entropy,
            "lempel_ziv": utils.get_lempel_ziv_entropy,
            "kontoyiannis": utils.get_kontoyiannis_entropy,
        }

        self.root = None
        self.feature_importances = None
        self.labels_universe = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        criterion: str = 'gini',
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int = None,
    ) -> Self:
        """
        Set the parameters for the Decision Tree Classifier.
        
        Parameters:
            - criterion (str): The criterion to use to compute the impurity.
            - max_depth (int): The maximum depth of the tree.
            - min_samples_split (int): The minimum number of samples required to split an internal node.
            - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            - max_features (int): The number of features to consider when looking for the best split.
        
        Returns:
            - self: The instance of the class.
        """
        self.params = {
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
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
    def get_impurity(
        self, 
        target_vector: np.ndarray,
        criterion: str
    ) -> float:
        """
        This function computes the impurity of a node.
        
        Parameters:
            - target_vector (np.array): The target values of the node.
            - criterion (str): The criterion to use to compute the impurity.
        
        Returns:
            - float: The impurity of the node.
        """
        if criterion in self.available_entropies:
            impurity = self.available_entropies[criterion](target_vector)
        
        else:
            raise ValueError(f"Unknown criterion : {criterion}")

        return impurity
    
    #?____________________________________________________________________________________ #
    def test_split(
        self, 
        feature: np.ndarray, 
        y_sorted: np.ndarray, 
        threshold: float, 
        nb_labels: int, 
        parent_impurity: float
    ) -> tuple:
        """
        Compute the information gain of a split.
        
        Parameters:
            - feature (np.ndarray): The feature to split on.
            - y_sorted (np.ndarray): The target values sorted according to the feature.
            - threshold (float): The threshold to split the feature.
            - nb_labels (int): The number of labels in the target vector.
            - parent_impurity (float): The impurity of the parent node.
        
        Returns:
            - tuple: A tuple containing the information gain and the threshold.
        """
        # ======= I. Initialize the variables ======= 
        left_mask = feature <= threshold
        right_mask = ~left_mask
        
        # ======= II. Check if the split is valid ======= 
        if np.sum(left_mask) < self.params['min_samples_leaf'] or np.sum(right_mask) < self.params['min_samples_leaf']:
            return (-1, threshold)
        
        # ======= III. Compute the impurities and the information gain ======= 
        left_impurity = self.get_impurity(y_sorted[left_mask], self.params['criterion'])
        right_impurity = self.get_impurity(y_sorted[right_mask], self.params['criterion'])
        
        # ======= IV. Compute the information gain ======= 
        child_impurity = (np.sum(left_mask) / nb_labels) * left_impurity + (np.sum(right_mask) / nb_labels) * right_impurity
        information_gain = parent_impurity - child_impurity
        
        return (information_gain, threshold)
    
    #?____________________________________________________________________________________ #
    def get_best_split(
        self, 
        features_matrix: np.ndarray, 
        target_vector: np.ndarray, 
        features_indexes: list
    ) -> tuple:
        """
        Compute the best split for the given features and target vector.
        
        Parameters:
            - features_matrix (np.ndarray): The features matrix.
            - target_vector (np.ndarray): The target vector.
            - features_indexes (list): The indexes of the features to consider for the split.
        
        Returns:
            - tuple: A tuple containing the best feature index, the best threshold, and the best information gain.
        """
        # ======= I. Initialize the variables ======= 
        best_gain = -1
        split_feature, split_threshold = None, None
        parent_impurity = self.get_impurity(target_vector, self.params['criterion'])
        nb_labels = len(target_vector)

        # ======= II. Precompute sorted features ======= 
        sorted_features = {}
        for feature_idx in features_indexes:
            feature = features_matrix[:, feature_idx]
            sorted_idx = np.argsort(feature)
            sorted_features[feature_idx] = (feature[sorted_idx], target_vector[sorted_idx])
        
        # ======= III. Define the Process Feature function ======= 
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
        
        # ======= IV. Process the features in parallel ======= 
        feature_results = Parallel(n_jobs=self.n_jobs)(delayed(process_feature)(feature_idx) for feature_idx in features_indexes)

        # ======= V. Check each result for best gain ======= 
        for result in feature_results:
            if result is None:
                continue
            feature_idx, thresholds_results = result
            
            if thresholds_results is None:
                continue  # Skip if no valid results
            
            for information_gain, threshold in thresholds_results:
                if information_gain > best_gain:
                    best_gain = information_gain
                    split_feature, split_threshold = feature_idx, threshold

        return split_feature, split_threshold, best_gain
    
    #?____________________________________________________________________________________ #
    def build_tree(
        self, 
        features_matrix: np.ndarray, 
        target_vector: np.ndarray, 
        depth=0
    ) -> Node:
        """
        Build the decision tree recursively.
        
        Parameters:
            - features_matrix (np.ndarray): The features matrix.
            - target_vector (np.ndarray): The target vector.
            - depth (int): The current depth of the tree.
        
        Returns:
            - Node: The root node of the decision tree.
        """
        # ======= I. Initialize the variables ======= 
        nb_samples, nb_features = features_matrix.shape
        num_labels = len(np.unique(target_vector))
        impurity = self.get_impurity(target_vector, self.params['criterion'])

        # ======= II. Check Stopping Criteria ======= 
        if (depth >= self.params['max_depth'] or num_labels == 1 or nb_samples < self.params['min_samples_split']):
            leaf_value = pd.Series(target_vector).value_counts().idxmax()
            leaf_samples = count_occurrences(universe=self.labels_universe, series=pd.Series(target_vector))
            node = Node(value=leaf_value, samples=leaf_samples, impurity=impurity)
            return node

        # ======= III. Get a random subset of the features ======= 
        max_features = min(nb_features, self.params['max_features']) if self.params['max_features'] else nb_features
        features_subset_indexes = np.random.choice(nb_features, max_features, replace=False)

        # ======= IV. Get the best split ======= 
        best_feature, best_threshold, best_gain = self.get_best_split(features_matrix, target_vector, features_subset_indexes)
        
        # ======= V. Check if the current split can't be improved ======= 
        if best_gain == -1:
            # If no good split, return leaf node
            leaf_value = pd.Series(target_vector).value_counts().idxmax()
            leaf_samples = pd.Series(target_vector).groupby(target_vector).count().tolist()
            node = Node(value=leaf_value, samples=leaf_samples, impurity=impurity)
            return node

        # ======= VI. Split the data and build the subtrees ======= 
        left_mask = features_matrix[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self.build_tree(features_matrix[left_mask], target_vector[left_mask], depth + 1)
        right_subtree = self.build_tree(features_matrix[right_mask], target_vector[right_mask], depth + 1)
        
        # ======= VII. Return the decision node ======= 
        samples = count_occurrences(universe=self.labels_universe, series=pd.Series(target_vector))
        node = Node(best_feature, best_threshold, left_subtree, right_subtree, samples=samples, impurity=impurity)

        return node

    #?____________________________________________________________________________________ #
    def traverse_tree(
        self, 
        row: np.ndarray, 
        node: Node
    ) -> Union[int, float]:
        """
        Traverse the tree to find the predicted value for a given row.
        
        Parameters:
            - row (np.ndarray): The row to predict.
            - node (Node): The current node in the tree.
        
        Returns:
            - Union[int, float]: The predicted value for the row.
        """
        # ======= I. Check if we reached a leaf node =======
        if node.is_leaf_node():
            return node.value
        
        # ======= II. Check if we need to go left or right =======
        elif row[node.feature] <= node.threshold:
            # We go left
            return self.traverse_tree(row, node.left)
        
        else:
            # We go right
            return self.traverse_tree(row, node.right)
    
    #?____________________________________________________________________________________ #
    def get_features_importances(
        self, 
        features_matrix: np.ndarray, 
        target_vector: np.ndarray
    ) -> np.ndarray:
        """
        Compute the feature importances of the decision tree.
        
        Parameters:
            - features_matrix (np.ndarray): The features matrix.
            - target_vector (np.ndarray): The target vector.
        
        Returns:
            - np.ndarray: The feature importances.
        """

        # ======= 0. Define the recursive function ======= 
        def compute_importance(node, total_samples):
            # Base case: we reached a leaf node
            if node is None or node.is_leaf_node():
                return

            # Update the importance of the feature used to split the node
            left_samples = np.sum(features_matrix[:, node.feature] <= node.threshold)
            right_samples = total_samples - left_samples

            self.features_importances[node.feature] += left_samples + right_samples
            compute_importance(node.left, left_samples)
            compute_importance(node.right, right_samples)

        # ======= I. Initialize the feature importances ======= 
        self.features_importances = np.zeros(features_matrix.shape[1])
        
        # ======= II. Compute the feature importances ======= 
        compute_importance(self.root, len(target_vector))
        
        # ======= III. Normalize the feature importances ======= 
        total_importance = np.sum(self.features_importances)
        
        if total_importance > 0 and not np.isnan(total_importance):
            self.features_importances /= total_importance
        else:
            # Log the issue for debugging purposes
            print("[WARNING] Feature importances sum to zero or NaN. Returning zero vector.")
            self.features_importances = np.zeros_like(self.features_importances)
        
        return self.features_importances
    
    #?_____________________________ User Functions _______________________________________ #
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> np.ndarray:
        """
        Fit the decision tree classifier to the training data.
        
        Parameters:
            - X_train (pd.DataFrame): The training data features.
            - y_train (pd.Series): The training data labels.
        
        Returns:
            - np.ndarray: The feature importances.
        """
        # ======= I. Process the data ======= 
        self.labels_universe = sorted(pd.Series(y_train).unique().tolist())
        
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Build the tree ======= 
        self.root = self.build_tree(X, y)
        
        # ======= III. Compute the key model statistics ======= 
        features_importances = self.get_features_importances(X, y)
        self.feature_importances = pd.Series(features_importances, index=X_train.columns)
        
        return self

    #?____________________________________________________________________________________ #
    def predict(
        self, 
        features_matrix: pd.DataFrame
    ) -> np.ndarray:
        """
        Classify the input data using the decision tree classifier.
        
        Parameters:
            - features_matrix (pd.DataFrame): The input data to classify.
        
        Returns:
            - np.ndarray: The predicted labels for the input data.
        """
        predictions = np.array([self.traverse_tree(row, self.root) for row in np.array(features_matrix)])
        
        return predictions


      
#! ==================================================================================== #
#! ================================ Helper Functions ================================== #
def count_occurrences(
    universe: list, 
    series: pd.Series
) -> list:
    """
    Count the occurrences of each value in the universe in the series.
    
    Parameters:
        - universe (list): The list of values to count.
        - series (pd.Series): The series to count the values in.
    
    Returns:
        - list: A list of counts for each value in the universe.
    """
    counts = series.value_counts().to_dict()
    occurences = [counts.get(val, 0) for val in universe]
    
    return occurences

