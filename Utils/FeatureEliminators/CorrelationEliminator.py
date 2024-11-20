import numpy as np
from FeatureEliminator import FeatureEliminator

class CorrelationEliminator(FeatureEliminator):
    """
    Keeps features that have an absolute correlation higher than a specified threshold
    with the target variable.
    """

    def __init__(self, X, y, correlation_threshold=0.1, num_features=None):
        """
        Initialize the CorrelationEliminator with a correlation threshold.

        Parameters:
        - X : Feature array. numpy ndarray.
        - y : Target array. numpy ndarray.
        - correlation_threshold : Minimum absolute correlation with the target required to keep a feature.
        - num_features : Number of top features to keep. 
        """
        self.correlation_threshold = correlation_threshold
        self.num_features = num_features
        super().__init__(X, y)

    def select_features(self):
        """
        Select features based on their absolute correlation with the target variable.
        """
        feature_means = np.mean(self.X, axis=0)
        target_mean = np.mean(self.y)
        feature_std = np.sqrt(np.sum((self.X - feature_means) ** 2, axis=0))
        target_std = np.sqrt(np.sum((self.y - target_mean) ** 2))

        y_reshaped = self.y.reshape(-1, 1) # reshape for substracting mean
        numerator = np.sum((self.X - feature_means) * (y_reshaped - target_mean), axis=0)
        denominator = feature_std * target_std
        denominator[denominator == 0] = 1000000 # avoid division by zero. Make it zero if denominator is 0.

        # Compute the correlation coefficients
        feature_correlations = numerator / denominator

        if self.num_features is not None:
            # Select top `num_features` based on absolute correlation
            sorted_indices = np.argsort(-np.abs(feature_correlations))
            self.feature_indices = sorted_indices[:self.num_features]
            self.feature_mask = np.zeros(self.X.shape[1], dtype=bool)
            self.feature_mask[self.feature_indices] = True
        elif self.correlation_threshold is not None:
            # Select features with absolute correlation above the threshold
            self.feature_mask = np.abs(feature_correlations) > self.correlation_threshold
            self.feature_indices = [i for i, keep in enumerate(self.feature_mask) if keep]