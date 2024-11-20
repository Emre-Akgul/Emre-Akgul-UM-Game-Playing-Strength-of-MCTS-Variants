import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the X and y into training and testing.
    """

    # Shuffle the data
    shuffled_indices = X.sample(frac=1, random_state=random_state).index
    X_shuffled = X.loc[shuffled_indices]
    y_shuffled = y.loc[shuffled_indices]
    
    # Calculate 
    test_size_count = int(test_size * len(X))
    
    # Split data
    X_train = X_shuffled.iloc[:-test_size_count]
    X_test = X_shuffled.iloc[-test_size_count:]
    y_train = y_shuffled.iloc[:-test_size_count]
    y_test = y_shuffled.iloc[-test_size_count:]
    
    return X_train, X_test, y_train, y_test

def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """
    Calculate root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

