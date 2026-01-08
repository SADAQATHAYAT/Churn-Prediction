import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates the model and prints metrics.
    """
    print(f"\n--- Evaluation for {model_name} ---")
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred

def plot_confusion_matrix(y_test, y_pred, model_name, save_path=None):
    """
    Plots the confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show() # Commented out for automated pipeline execution

def explain_model_shap(model, X_train, X_test, feature_names, model_name):
    """
    Generates SHAP summary plot.
    """
    print(f"Generating SHAP explanation for {model_name}...")
    
    # SHAP depends on model type
    try:
        if model_name == 'XGBoost':
            explainer = shap.TreeExplainer(model)
        else: 
            # Generic KernelExplainer or TreeExplainer for RF
            # TreeExplainer is faster for Trees
            explainer = shap.TreeExplainer(model)
            
        # Compute SHAP values for a subset of test data for speed
        shap_values = explainer.shap_values(X_test[:100])
        
        # Plot
        plt.figure()
        # Handle different SHAP return types (some return list of arrays for classification)
        if isinstance(shap_values, list):
             shap.summary_plot(shap_values[1], X_test[:100], feature_names=feature_names, show=False)
        else:
             shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
             
        plt.title(f"SHAP Summary - {model_name}")
        plt.tight_layout()
        plt.savefig(f"shap_summary_{model_name}.png")
        print(f"SHAP summary saved as shap_summary_{model_name}.png")
        
    except Exception as e:
        print(f"Could not generate SHAP plots: {e}")
