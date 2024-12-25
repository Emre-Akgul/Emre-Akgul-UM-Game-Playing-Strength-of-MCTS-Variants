import numpy as np
import pandas as pd
from .FeatureEliminator import FeatureEliminator
from collections import Counter

class MutualInformationEliminator(FeatureEliminator):
    """
    Feature eliminator based on mutual information between features and target.
    Keeps features with mutual information scores above a threshold.
    """
    
    def __init__(self, X, y, threshold=None, num_features=None):
        """
        Initialize the MutualInformationEliminator.
        
        Parameters:
        - X : numpy.ndarray
            Feature array
        - y : numpy.ndarray
            Target array
        - threshold : float, default=0.01
            Minimum mutual information score to keep a feature
        """
        if num_features is not None and threshold is not None:
            raise ValueError("Only one of 'threshold' and 'num_features' can be provided")

        if threshold is None and num_features is None:
            raise ValueError("Either 'threshold' or 'num_features' must be provided")
        
        self.threshold = threshold
        self.num_features = num_features
        self.X_df = pd.DataFrame(X)
        super().__init__(X, y)
    
    # Modified version of select_features method
    def select_features(self):
        mi_scores = self.calculate_mutual_information(self.X_df, self.y)
        
        if self.num_features is not None:
            selected_features = mi_scores.index[:self.num_features]
        else:
            selected_features = mi_scores[mi_scores["Mutual Information"] > self.threshold].index
        
        self.feature_indices = selected_features.values
        self.feature_mask = np.zeros(self.X.shape[1], dtype=bool)
        self.feature_mask[self.feature_indices] = True

    def calculate_mutual_information(self, X, y):
        def entropy(values):
            """Calculate the entropy of a dataset."""
            # For continuous variables, use binning
            if values.dtype.kind in ['f', 'i']:
                values = pd.qcut(values, q=20, labels=False, duplicates='drop')
            
            total = len(values)
            counts = Counter(values)
            probabilities = np.array([count / total for count in counts.values()])
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        def conditional_entropy(feature, target):
            """Calculate the conditional entropy of target given feature."""
            if feature.dtype.kind in ['f', 'i']:
                feature = pd.qcut(feature, q=20, labels=False, duplicates='drop')
                
            total = len(feature)
            unique_values = np.unique(feature)
            cond_entropy = 0
            
            for value in unique_values:
                indices = np.where(feature == value)[0]
                subset = target[indices]
                prob = len(indices) / total
                cond_entropy += prob * entropy(subset)
                
            return cond_entropy
        
        # Input validation
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or numpy array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
            
        # Drop categorical columns and store column names
        numeric_X = X.select_dtypes(include=['int64', 'float64'])
        colnames = numeric_X.columns
        
        # Convert to numpy arrays
        X_values = numeric_X.values
        y_values = y.values if isinstance(y, pd.Series) else y
        
        # Calculate mutual information for each feature
        mutual_info = []
        target_entropy = entropy(y_values)
        
        for i in range(X_values.shape[1]):
            feature = X_values[:, i]
            cond_entropy = conditional_entropy(feature, y_values)
            mi = target_entropy - cond_entropy
            mutual_info.append(mi)
        
        # Create a DataFrame with results
        mutual_info_df = pd.DataFrame({
            "Feature": colnames,
            "Mutual Information": mutual_info
        }).sort_values(by="Mutual Information", ascending=False)
        
        return mutual_info_df