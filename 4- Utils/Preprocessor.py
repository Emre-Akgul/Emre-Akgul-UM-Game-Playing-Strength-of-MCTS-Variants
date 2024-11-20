import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self, normalize=False, standardize=False):
        """
        Initialize the Preprocessor. Takes pandas dataframe, normalizes and standardizes it. Return numpy array.

        Parameters:
        - normalize: Normalize if true.
        - standardize: Standardize if true.
        """
        self.normalize = normalize
        self.standardize = standardize

        self.feature_min = None
        self.feature_max = None
        self.feature_mean = None
        self.feature_std = None

    def fit(self, df):
        """
        Fit the preprocessor to the data.
        Extracts feature_min, feature_max for normalization.
        Extracts feature_mean, feature_std for standardization

        Parameters:
        - df: A pandas DataFrame to normalize or standardize.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame.")

        if self.normalize:
            self.feature_min = df.min(axis=0)
            self.feature_max = df.max(axis=0)

        if self.standardize:
            if self.normalize:
                normalized_df = (df - self.feature_min) / (self.feature_max - self.feature_min)
                self.feature_mean = normalized_df.mean(axis=0)
                self.feature_std = normalized_df.std(axis=0)
            else:
                self.feature_mean = df.mean(axis=0)
                self.feature_std = df.std(axis=0)

    def transform(self, df):
        """
        Transform the data.

        Parameters:
        - df: A pandas DataFrame to be transformed.

        Returns:
        - A NumPy ndarray with the transformed data.
        """            

        if self.normalize:
            if self.feature_min is None or self.feature_max is None:
                raise ValueError("Not fitted yet.")
            normalized = (df - self.feature_min) / (self.feature_max - self.feature_min)
        else:
            normalized = df

        if self.standardize:
            if self.feature_mean is None or self.feature_std is None:
                raise ValueError("Not fitted yet.")
            standardized = (normalized - self.feature_mean) / self.feature_std
            return standardized.to_numpy()

        return normalized.to_numpy()

    def fit_transform(self, df):
        """
        Fit and transform the data.

        Parameters:
        - df: A pandas DataFrame to be fitted and transformed.

        Returns:
        - A NumPy ndarray with the transformed data.
        """
        self.fit(df)
        return self.transform(df)
