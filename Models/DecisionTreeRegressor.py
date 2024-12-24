import pandas as pd
import numpy as np
from Models.Model import Model

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
    
class DecisionTreeRegressor(Model):
    def __init__(self, min_samples_split=20, max_depth= 100, n_features=None):
        """
        Initialize the DecisionTreeRegressor with minimal parameters.

        Parameters:
        max_depth: maximum depth of the tree. Integer or None. None is no bound.

        min_samples_split: int, optional (default=20) # statsquest
            The minimum number of samples required to split an internal node.
        """
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        Fit the decision tree to the data.

        Parameters:
        - X: Input value array for training data. Should be numpy array with shape (n_samples, n_features).
        - y: Target value array for training data. Should be numpy array with shape (n_samples, ).
        """
        total_n_features = X.shape[1]
        if self.n_features is None:
            self.n_features = total_n_features
        else:
            self.n_features = min(self.n_features, total_n_features)

        self.root = self._grow_tree(X, y)
         
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        feat_indices = np.random.choice(n_features, self.n_features, replace=False)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, feat_indices)

        # If no valid split, return leaf node
        if best_feature is None or best_threshold is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        # Create child nodes
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(best_feature, best_threshold, left, right)


    def _best_split(self, X, y, feat_indices):
        best_gain = float('-inf')
        split_idx, split_threshold = None, None

        for feat_id in feat_indices:
            X_column = X[:, feat_id]
            sorted_indices = np.argsort(X_column)
            X_column, y_sorted = X_column[sorted_indices], y[sorted_indices]

            # Evaluate midpoints between consecutive values
            for i in range(1, len(y)):
                if X_column[i] == X_column[i - 1]:
                    continue
                threshold = (X_column[i] + X_column[i - 1]) / 2
                gain = self._information_gain(y_sorted, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_id
                    split_threshold = threshold

        # Handle case where no split is found
        if split_idx is None or split_threshold is None:
            return None, None

        return split_idx, split_threshold


    
    def _information_gain(self, y, X_column, split_threshold):
        # Ensure y is a numpy array
        if isinstance(y, pd.Series):
            y = y.values

        # Parent variance
        parent_variance = np.var(y)

        left_indices, right_indices = self._split(X_column, split_threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        # Calculate variance for children
        n = len(y)
        n_l, n_r = len(left_indices), len(right_indices)
        left_variance = np.var(y[left_indices])
        right_variance = np.var(y[right_indices])
        weighted_child_variance = (n_l / n) * left_variance + (n_r / n) * right_variance

        # Return variance reduction as information gain
        return parent_variance - weighted_child_variance

    
    def _split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices

    def predict(self, X):
        """
        Predict the target values for given inputs.

        Parameters:
        - X: Input value array for prediction. Should be numpy array with shape (n_samples, n_features).

        Returns:
        - y: Predictions values for input array X. numpy array with shape (n_samples, )
        """
        
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
