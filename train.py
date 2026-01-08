import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

class ModelTrainer:
    """
    Class to handle model training and hyperparameter tuning.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        }
    
    def tune_hyperparameters(self, model_name, X_train, y_train):
        """
        Performs RandomizedSearchCV for the specified model.
        """
        print(f"Tuning hyperparameters for {model_name}...")
        
        if model_name == 'RandomForest':
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        elif model_name == 'XGBoost':
            param_dist = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        else:
            raise ValueError(f"Model {model_name} not supported.")
            
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        scorer = make_scorer(f1_score)
        
        random_search = RandomizedSearchCV(
            self.models[model_name], 
            param_distributions=param_dist, 
            n_iter=5, # Low iterations for speed in this demo
            scoring=scorer, 
            cv=kf, 
            verbose=1, 
            n_jobs=-1,
            random_state=self.random_state
        )
        
        random_search.fit(X_train, y_train)
        print(f"Best parameters for {model_name}: {random_search.best_params_}")
        return random_search.best_estimator_

    def train_model(self, model_name, X_train, y_train, best_model=None):
        """
        Trains the model.
        """
        if best_model:
            model = best_model
        else:
            model = self.models[model_name]
            
        model.fit(X_train, y_train)
        return model
