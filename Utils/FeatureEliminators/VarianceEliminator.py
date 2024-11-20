import numpy as np

from .FeatureEliminator import FeatureEliminator
class VarianceEliminator(FeatureEliminator):
    """
    Eliminates features with variance below a specified threshold.
    """

    def __init__(self, X, y, threshold=0.01):
        """
        Initialize the VarianceEliminator with a variance threshold.

        Parameters:
        - X : Feature array. numpy ndarray.
        - y : Target array. numpy ndarray.
        - threshold : Minimum variance required to retain a feature.
        """
        self.threshold = threshold
        super().__init__(X, y)

    def select_features(self):
        """
        Select features with variance above the specified threshold.
        """
        feature_means = np.mean(self.X, axis=0)
        feature_variances = np.sum((self.X - feature_means) ** 2, axis=0) / self.X.shape[0]
        
        self.feature_mask = feature_variances > self.threshold
        self.feature_indices = [i for i, is_selected in enumerate(self.feature_mask) if is_selected]