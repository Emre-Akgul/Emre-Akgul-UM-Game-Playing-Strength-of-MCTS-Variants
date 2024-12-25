import numpy as np
from Models.Model import Model
from Models.DecisionTreeRegressor import DecisionTreeRegressor

class GradientBoostingRegressor(Model):
    def __init__(self, learning_rate, n_estimators, max_depth, min_samples_split):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.base_prediction = None
    
    def fit(self, X, y):
        # Start with mean prediction
        self.base_prediction = np.mean(y)
        
        y_pred = np.full(np.shape(y), self.base_prediction)

        for i in range(self.n_estimators):
            # Compute residuals (gradient of RMSE loss)
            residual = y - y_pred
            
            # Fit a decision tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
            self.trees.append(tree)
            
            # gradient weak learner addition
            update = tree.predict(X)
            y_pred += self.learning_rate * update

    def predict(self, X):
        # base prediction
        predictions = np.full(X.shape[0], self.base_prediction)
        
        # gradient boosting
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            
        return predictions
