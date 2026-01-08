import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_features(df):
    """
    Creates new features from existing ones.
    """
    df = df.copy()
    
    # Feature 1: Ratio of TotalCharges to Tenure (should approximate MonthlyCharges, but variations are interesting)
    # Handle division by zero for new customers (tenure=0)
    df['AverageCharges'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    
    # Feature 2: Is the customer a long-term contract user with high charges?
    # This might interact with churn.
    
    # Handling Binary Columns to numeric for correlation analysis later
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        
    # Drop rows where TotalCharges is NaN (generated in our script)
    # Alternatively we could fill it, but dropping is safer for this exercise if small amount
    df = df.dropna(subset=['TotalCharges'])
    
    return df

def feature_selection_correlation(df, target_col, threshold=0.9):
    """
    Drops highly correlated features to reduce multicollinearity.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Features to drop due to high correlation: {to_drop}")
    return df.drop(columns=to_drop)

def plot_correlation(df):
    """
    Plots correlation heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    # In a real app we might save this plot
    # plt.savefig(f"{ARTIFACTS_PATH}/correlation_matrix.png")
