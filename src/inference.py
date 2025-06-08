# src/inference.py

import sys
import pandas as pd
from data_processing import load_and_merge
from feature_engineering import build_features
from model import load_model

def predict_deals(
    won_path: str, lost_path: str, activities_path: str, model_path: str
):
    # 1) Load & merge input data
    merged = load_and_merge(won_path, lost_path, activities_path)
    # 2) Build features (but drop 'outcome', since these are “unlabeled” in real inference)
    features_df = build_features(merged)
    X = features_df.drop(columns=['outcome'])

    # 3) Load model
    model = load_model(model_path)

    # 4) Predict probabilities
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    # 5) Return a DataFrame with deal_id, prediction, probability
    results = pd.DataFrame({
        "deal_id": merged['deal_id'],
        "won_probability": proba,
        "won_prediction": preds
    })
    return results

if __name__ == "__main__":
    # Example usage:
    # python inference.py data/won_deals.json data/lost_deals.json data/activities.json models/XGB_model.pkl
    if len(sys.argv) != 5:
        print("Usage: python inference.py <won.json> <lost.json> <activities.json> <model.pkl>")
        sys.exit(1)
    won_path, lost_path, activities_path, model_path = sys.argv[1:]
    preds_df = predict_deals(won_path, lost_path, activities_path, model_path)
    print(preds_df.head())
    # Optionally: save to CSV
    preds_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
