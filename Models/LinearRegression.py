import numpy as np
from .Model import Model

class LinearRegression(Model):
    def __init__(self, fit_method='ols', loss_function="rmse", l1=0, l2=0, learning_rate=0.01, epochs=1000, min_step_size=0.001, gradient_descent='batch', batch_size=32):
        super().__init__()

        # general parameters
        self.fit_method = fit_method
        self.loss_function = loss_function

        # regularization parameters
        self.l1 = l1
        self.l2 = l2

        # gradient descent parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_step_size = min_step_size
        self.gradient_descent = gradient_descent
        self.batch_size = batch_size

        # initialize weights to none
        self.weights = None # W0 will be bias.

    def fit(self, X, y):
        # Add bias terms coefficent to the X for easier bias term handling.
        X = np.c_[np.ones((X.shape[0], 1)), X]

        if self.fit_method == 'ols':
            self._fit_ols(X, y)
        elif self.fit_method == 'gd':
            self._fit_gd(X, y)
        else:
            raise ValueError("fit_method should be either 'ols' or 'gd'")


    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Add bias terms coefficent to the X for prediction.
        X = np.c_[np.ones((X.shape[0], 1)), X]

        y = self._predict(X)
        return y
    
    def _calculate_gradient(self, X, y):
        y_pred = self._predict(X)

        if self.loss_function == 'rmse':
            loss_gradient = - X.T @ (y - y_pred) / (X.shape[0] * np.sqrt(np.mean((y - y_pred) ** 2))) + self.l1 * np.sign(self.weights) + 2 * self.l2 * self.weights - self.l1 * np.sign(self.weights[0]) - 2 * self.l2 * self.weights[0]
        else:
            raise ValueError("loss_function should be rmse.")

        return loss_gradient

    def _fit_ols(self, X, y):
        """
        Fit the model to the data using ordinary least squares fit method by calculating weights by given formula.
        """

        self.weights = np.linalg.pinv(X.T @ X + self.l2 * np.identity(X.shape[1])) @ X.T @ y

    def _fit_gd(self, X, y):
        if self.gradient_descent == 'batch':
            self._fit_gd_batch(X, y)
        elif self.gradient_descent == 'stochastic':
            self._fit_gd_stochastic(X, y)
        elif self.gradient_descent == 'mini-batch':
            self._fit_gd_mini_batch(X, y)
        else:
            raise ValueError("Incorrect gradient_descent value. Possible values: batch, stochastic, mini-batch.")

    def _fit_gd_batch(self, X, y):
        """
        Fit the model to the data using batch gradient descent method by updating weights untill convergence.
        Batch gradients use all the training data for updating weights at each step.
        """

        # Initialize weights
        self.weights = np.random.randn(X.shape[1], ) * 0.01
        self.weights[0] = 0 # Thats what they do in NN
        
        # Gradient Descent Loop
        for _ in range(self.epochs):
            gradient = self._calculate_gradient(X, y)
            self.weights = self.weights - self.learning_rate * gradient

    def _fit_gd_stochastic(self, X, y):
        """
        Fit the model to the data using batch gradient descent method by updating weights untill convergence.
        Stochastic gradients use all the training data for updating weights at each step.
        """

        self.batch_size = 1
        self._fit_gd_mini_batch(X, y)

    def _fit_gd_mini_batch(self, X, y):
        """
        Fit the model to the data using batch gradient descent method by updating weights untill convergence.
        Batch gradients use all the training data for updating weights at each step.
        """

        # Initialize weights
        self.weights = np.random.randn(X.shape[1], ) * 0.01
        self.weights[0] = 0 # Thats what they do in NN

        n = X.shape[0]
        current_index = 0

        for epoch in range(self.epochs):
            if epoch % n == 0:
                indices = np.arange(n)
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]
            
            current_X, current_y = X[current_index : min(current_index + self.batch_size, n)], y[current_index : min(current_index + self.batch_size, n)]
            current_index = min(current_index + self.batch_size, n) % n
            gradient = self._calculate_gradient(current_X, current_y)
            self.weights = self.weights - self.learning_rate * gradient

    def _predict(self, X):
        """
        Helper method for gradient descent. Using self.predict add 1s for the biases.
        """
        return X @ self.weights