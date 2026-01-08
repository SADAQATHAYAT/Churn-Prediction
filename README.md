# Customer Churn Prediction Project

This project implements an end-to-end Machine Learning pipeline to predict customer churn for a telecommunications company.

## Project Structure

- `data/`: Contains the dataset (generated synthetically).
- `src/`: Source code for the pipeline.
    - `data_loader.py`: Data loading and inspection.
    - `preprocessing.py`: Encoding, scaling, and SMOTE.
    - `feature_engineering.py`: Feature creation and selection.
    - `train.py`: Model training and hyperparameter tuning.
    - `evaluation.py`: Metrics, confusion matrix, and SHAP.
    - `config.py`: Configuration constants.
- `main.py`: Orchestration script to run the full pipeline.
- `generate_data.py`: Script to generate synthetic data.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate Data (if not already present):
   ```bash
   python generate_data.py
   ```

## Usage

Run the entire pipeline:
```bash
python main.py
```

## Features

- **Data Generation**: Creates a realistic dataset with 50,000+ records.
- **Preprocessing**: Handles missing values, performs One-Hot Encoding and Standard Scaling.
- **Imbalance Handling**: Uses SMOTE to balance the training set.
- **Feature Engineering**: Creates derived features like `AverageCharges`.
- **Modeling**: Trains Random Forest and XGBoost models.
- **Explainability**: Uses SHAP values to explain model predictions.

## Results

Metrics and plots (Confusion Matrix, SHAP Summary) vary based on the random generation but typically achieve robust F1-scores.
