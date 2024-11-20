import numpy as np
from FeatureEliminator import FeatureEliminator
from ...Models.LinearRegression import LinearRegression

class LassoFeatureEliminator(FeatureEliminator):
    """
    Feature eliminator using Lasso (L1 regularization) regression. 
    Features with coefficients close to zero are eliminated.
    """

    def __init__(self, X, y, l1=0.1, threshold=1e-5):
        """
        Initialize the LassoFeatureEliminator.

        Parameters:
        - X : Feature array. numpy ndarray.
        - y : Target array. numpy ndarray.
        - l1 : L1 regularization strength. Controls sparsity.
        - threshold : Threshold below which feature coefficients are considered zero.
        """
        self.l1 = l1
        self.threshold = threshold
        super().__init__(X, y)

    def select_features(self):
        """
        Select features based on Lasso regression coefficients.
        Features with coefficients close to zero are eliminated.
        """
        # Fit a Lasso regression model
        lasso_model = LinearRegression(fit_method='gd', l1=self.l1, loss_function="rmse", epochs=10000, learning_rate=0.01) # high iteration low learning rate my favorite
        lasso_model.fit(self.X, self.y)

        # Extract coefficients
        coefficients = lasso_model.weights[1:] # 1: beacuse first one is bias

        # Identify features with coefficients above the threshold
        self.feature_mask = np.abs(coefficients) > self.threshold
        self.feature_indices = [i for i, keep in enumerate(self.feature_mask) if keep]