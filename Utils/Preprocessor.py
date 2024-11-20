import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self, normalize=False, standardize=False, one_hot_encode=False):
        """
        Initialize the Preprocessor. Takes pandas dataframe, normalizes and standardizes it. Return numpy array.

        Parameters:
        - normalize: Normalize if true.
        - standardize: Standardize if true.
        - one_hot_endoce: One hot encode if true.
        """
        self.normalize = normalize
        self.standardize = standardize
        self.one_hot_encode = one_hot_encode

        self.feature_min = None
        self.feature_max = None
        self.feature_mean = None
        self.feature_std = None

        self.categorical_columns = None
        self.categories_per_column = {}

    def fit(self, df):
        """
        Fit the preprocessor to the data.
        Extracts categories and categories_per_column for one-hot encoding.
        Extracts feature_min, feature_max for normalization.
        Extracts feature_mean, feature_std for standardization
        
        Parameters:
        - df: A pandas DataFrame to normalize or standardize.
        """

        if self.one_hot_encode:
            self.categorical_columns = df.select_dtypes(include='category').columns.tolist()
            for col in self.categorical_columns:
                categories = df[col].cat.categories.tolist()
                self.categories_per_column[col] = categories

            # Perform one-hot encoding temporarily to add new columns for normalization/standardization
            one_hot_encoded_df = df.copy()
            for col in self.categorical_columns:
                categories = self.categories_per_column[col]
                one_hot_encoded_df[col] = pd.Categorical(one_hot_encoded_df[col], categories=categories)
                for category in categories[1:]:  # Dropping the first one for one-hot encoding
                    new_col = f"{col}_{category}"
                    one_hot_encoded_df[new_col] = (one_hot_encoded_df[col] == category).astype(int)
                one_hot_encoded_df.drop(columns=col, inplace=True)

            df_num = one_hot_encoded_df
        else:
            df_num = df

        if self.normalize:
            self.feature_min = df_num.min(axis=0)
            self.feature_max = df_num.max(axis=0)

        if self.standardize:
            if self.normalize:
                normalized_df = (df_num - self.feature_min) / (self.feature_max - self.feature_min)
                self.feature_mean = normalized_df.mean(axis=0)
                self.feature_std = normalized_df.std(axis=0)
            else:
                self.feature_mean = df_num.mean(axis=0)
                self.feature_std = df_num.std(axis=0)

    def transform(self, df):
        """
        Transform the data.

        Parameters:
        - df: A pandas DataFrame to be transformed.

        Returns:
        - A NumPy ndarray with the transformed data.
        """            

        df_transformed = df.copy()

        if self.one_hot_encode:
            if self.categorical_columns is None or self.categories_per_column is None:
                raise ValueError("Not fitted yet.")
            for col in self.categorical_columns:
                categories = self.categories_per_column[col]
                df_transformed[col] = pd.Categorical(df_transformed[col], categories=categories)
                for category in categories[1:]: # Dropping the first one.
                    new_col = f"{col}_{category}"
                    df_transformed[new_col] = (df_transformed[col] == category).astype(int)
                df_transformed.drop(columns=col, inplace=True)

        df_num = df_transformed
        self.column_names = df_num.columns

        if self.normalize:
            if self.feature_min is None or self.feature_max is None:
                raise ValueError("Not fitted yet.")
            normalized = (df_num - self.feature_min) / (self.feature_max - self.feature_min)
        else:
            normalized = df_num

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
    
    def get_column_names(self):
        return self.column_names
