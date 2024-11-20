from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    def __init__(self):
        """
        Initialize the Model.
        I hate python interfaces.
        """

    @abstractmethod
    def fit(self, X, y):
        """
        Method for fitting the model to data.
        Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Method for predicting values.
        Must be implemented by all subclasses.
        """
        pass