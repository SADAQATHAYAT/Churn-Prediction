import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE
from src.data_loader import load_data, inspect_data, check_outliers
from src.preprocessing import DataPreprocessor, handle_imbalance, encode_target
from src.feature_engineering import create_features as engineer_features
from src.feature_engineering import feature_selection_correlation
from src.train import ModelTrainer
from src.evaluation import evaluate_model, plot_confusion_matrix, explain_model_shap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    print("Starting Customer Churn Prediction Pipeline...\n")
    
    # 1. Load Data
    df = load_data(DATA_PATH)
    inspect_data(df)
    
    # 2. Feature Engineering (Creation)
    print("\n--- Feature Engineering ---")
    df = engineer_features(df)
    
    # 3. Split Data
    # Drop customerID as it is unique and causes memory issues during OHE
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    # Encode Target
    y_encoded, le_target = encode_target(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    # 4. Preprocessing
    print("\n--- Preprocessing ---")
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = DataPreprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names for SHAP
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"Feature_{i}" for i in range(X_train_processed.shape[1])]

    # 5. Handle Imbalance (SMOTE)
    print("\n--- Handling Class Imbalance (SMOTE) ---")
    X_train_resampled, y_train_resampled = handle_imbalance(X_train_processed, y_train)
    
    # 6. Modeling & Evaluation
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    # rf_model = trainer.tune_hyperparameters('RandomForest', X_train_resampled, y_train_resampled) # Commented for speed in demo, using default dict
    rf_model = trainer.train_model('RandomForest', X_train_resampled, y_train_resampled)
    
    evaluate_model(rf_model, X_test_processed, y_test, 'RandomForest')
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    # xgb_model = trainer.tune_hyperparameters('XGBoost', X_train_resampled, y_train_resampled) # Commented for speed in demo
    xgb_model = trainer.train_model('XGBoost', X_train_resampled, y_train_resampled)
    
    xgb_pred = evaluate_model(xgb_model, X_test_processed, y_test, 'XGBoost')
    
    # 7. Explainability
    print("\n--- Explainability ---")
    explain_model_shap(xgb_model, X_train_processed, X_test_processed, feature_names, 'XGBoost')
    
    print("\nPipeline Completed Successfully.")

if __name__ == "__main__":
    main()
