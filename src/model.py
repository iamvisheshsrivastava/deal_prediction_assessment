# src/model.py

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

def train_model(features_df: pd.DataFrame, model_path: str = "models/XGB_model.pkl"):
    """
    Trains an XGBoost classifier and saves the model to disk.
    Expects features_df with 'outcome' column as target and rest as features.
    """
    X = features_df.drop(columns=['outcome'])
    y = features_df['outcome']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Set up basic XGBoost with default params
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )

    # Optional: hyperparameter tuning (example)
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(clf, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Feature importance (optional)
    importances = best_model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("Top Features:\n", feat_importances.head(10))
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {model_path}")
    return best_model

def load_model(model_path: str = "models/XGB_model.pkl"):
    """
    Load a trained model from disk.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    # Quick test pipeline
    from data_processing import load_and_merge
    from feature_engineering import build_features
    
    merged_df = load_and_merge("data/won_deals.json",
                               "data/lost_deals.json",
                               "data/activities.json")
    features_df = build_features(merged_df)
    model = train_model(features_df, model_path="models/XGB_model.pkl")
