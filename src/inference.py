# src/inference.py

import sys
import pandas as pd
from data_processing import load_and_merge
from feature_engineering import build_features
from model import load_model

def predict_deals(
    won_path: str, lost_path: str, activities_path: str, model_path: str
):
    merged = load_and_merge(won_path, lost_path, activities_path)
    features_df = build_features(merged)
    X = features_df.drop(columns=['outcome'])

    model = load_model(model_path)

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    results = pd.DataFrame({
        "deal_id": merged['deal_id'],
        "won_probability": proba,
        "won_prediction": preds
    })
    return results

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python inference.py <won.json> <lost.json> <activities.json> <model.pkl>")
        sys.exit(1)
    won_path, lost_path, activities_path, model_path = sys.argv[1:]
    preds_df = predict_deals(won_path, lost_path, activities_path, model_path)
    print(preds_df.head())
    preds_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
