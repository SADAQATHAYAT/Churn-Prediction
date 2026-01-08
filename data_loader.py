import pandas as pd
import numpy as np
from src.config import DATA_PATH

def load_data(path=DATA_PATH):
    """
    Loads data from the specified path.
    """
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found at {path}. Please check the path and try again.")
        raise

def inspect_data(df):
    """
    Performs basic data inspection.
    """
    print("--- Data Info ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Duplicates ---")
    print(df.duplicated().sum())
    print("\n--- Statistics ---")
    print(df.describe())

def check_outliers(df, numeric_cols):
    """
    Checks for outliers in numeric columns using IQR method.
    """
    outlier_indices = []
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.extend(outliers)
        
    print(f"Potential outliers detected in {len(set(outlier_indices))} rows.")
    return list(set(outlier_indices))
