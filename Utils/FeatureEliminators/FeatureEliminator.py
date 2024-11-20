from abc import ABC, abstractmethod
import numpy as np

class FeatureEliminator(ABC):
    """
    FeatureEliminator interface.    

    Does feature elimination based on supervised or unsupervised methods.
    """

    def __init__(self, X, y):
        """
        Initialize the FeatureEliminator.
        X is feature array.
        y is 

        Parameterers:
        - X : Feature array. numpy ndarray.
        - y : Target array. numpy ndarray.
        """
        self.X = X
        self.y = y

        self.feature_indices = None
        self.feature_mask = None

        self.select_features()

    @abstractmethod
    def select_features(self):
        """        
        Select features based on some supervised or unsupervised method.
        Extracts feature indices and feature_maskç
        """
        pass

    def get_feature_indices(self):
        """
        Get indices of features to keep.
        
        Return numpy ndarray consist of indices to keep.
        """
        return self.feature_indices

    def get_feature_mask(self):
        """
        Get feature mask. Feature mask consist of True, False values.
        True for keeping the index, False for dropping.
        
        Return numpy ndarray mask.
        """
        return self.feature_mask

