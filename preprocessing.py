import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle missing values, scaling, and encoding.
    """
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.pipeline = None
        self.label_encoders = {}
        
    def fit(self, X, y=None):
        # Numeric Pipeline: Fill missing with median -> Scale
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Categorical Pipeline: OneHotEncode
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        self.pipeline.fit(X)
        return self
        
    def transform(self, X):
        return self.pipeline.transform(X)

    def get_feature_names_out(self):
        # Helper to get feature names after OneHotEncoding
        # This is strictly for preserving column names if needed
        if hasattr(self.pipeline, 'get_feature_names_out'):
             return self.pipeline.get_feature_names_out()
        return None

def handle_imbalance(X, y):
    """
    Applies SMOTE to handle class imbalance.
    """
    print(f"Original class distribution: {np.bincount(y)}")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Resampled class distribution: {np.bincount(y_res)}")
    return X_res, y_res

def encode_target(y):
    """
    Label encodes the target variable.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le
