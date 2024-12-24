import pandas as pd
import numpy as np
from Models.Model import Model
from Models.DecisionTreeRegressor import DecisionTreeRegressor
import random

class RandomForestRegressor(Model):
    def __init__(self, n_estimators=10, min_samples_split=20, max_depth= 100, n_features=None):
        """
        Initialize the DecisionTreeRegressor with minimal parameters.

        Parameters:
        max_depth: maximum depth of the tree. Integer or None. None is no bound.

        min_samples_split: int, optional (default=20) # statsquest
            The minimum number of samples required to split an internal node.
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []


    def fit(self, X, y):
        """
        Fit the decision tree to the data.

        Parameters:
        - X: Input value array for training data. Should be numpy array with shape (n_samples, n_features).
        - y: Target value array for training data. Should be numpy array with shape (n_samples, ).
        """

        if self.n_features is None:
            # if n_features is not set, use square root of total features
            self.n_features = int(np.sqrt(X.shape[1])) 
        else:
            self.n_features = min(self.n_features, X.shape[1])

        for _ in range(self.n_estimators):
            tree_max_depth = random.randint(max(self.max_depth - 3, 1), self.max_depth)  # Random max depth
            tree_min_samples_split = random.randint(self.min_samples_split * 0.5, self.min_samples_split * 1.5)

            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_features=self.n_features)

            bootstrap_indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the target values for given inputs.

        Parameters:
        - X: Input value array for prediction. Should be numpy array with shape (n_samples, n_features).

        Returns:
        - y: Predictions values for input array X. numpy array with shape (n_samples, )
        """
        
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += tree.predict(X)
        predictions /= len(self.trees)
        return predictions
    
