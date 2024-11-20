import numpy as np
from Models.Model import Model

class DecisionTreeRegressor(Model):
    def __init__(self, max_depth=None, min_samples_split=2):
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
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree to the data.

        Parameters:
        - X: Input value array for training data. Should be numpy array with shape (n_samples, n_features).
        - y: Target value array for training data. Should be numpy array with shape (n_samples, ).
        """
        pass

    def predict(self, X):
        """
        Predict the target values for given inputs.

        Parameters:
        - X: Input value array for prediction. Should be numpy array with shape (n_samples, n_features).

        Returns:
        - y: Predictions values for input array X. numpy array with shape (n_samples, )
        """
        pass
