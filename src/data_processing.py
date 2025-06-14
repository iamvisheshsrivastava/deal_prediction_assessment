import json
import pandas as pd
from typing import Tuple

def load_json(path: str) -> pd.DataFrame:
    """
    Load a JSON file and return a pandas DataFrame.
    Expects an array of JSON objects at top level.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}")
    return pd.DataFrame(data)

def validate_deals(df: pd.DataFrame, required_cols: list):
    """
    Ensure that required columns exist, no null IDs, etc.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in deals DataFrame: {missing}")
    if 'deal_id' in df.columns and df['deal_id'].isnull().any():
        raise ValueError("Null deal_id found in deals DataFrame")

def load_and_merge(
    won_path: str, lost_path: str, activities_path: str
) -> pd.DataFrame:
    """
    Load won_deals, lost_deals, and activities, then merge into one DataFrame.
    Returns a DataFrame with:
      - deal_id
      - outcome (1 = won, 0 = lost)
      - all activities (as a list of dicts) per deal
    """
    won_df = load_json(won_path)
    lost_df = load_json(lost_path)

    if 'id' in won_df.columns:
        won_df.rename(columns={'id': 'deal_id'}, inplace=True)
    if 'id' in lost_df.columns:
        lost_df.rename(columns={'id': 'deal_id'}, inplace=True)

    won_df['outcome'] = 1
    lost_df['outcome'] = 0

    deals_df = pd.concat([won_df, lost_df], ignore_index=True)
    
    validate_deals(deals_df, required_cols=['deal_id', 'outcome'])
    
    act_df = load_json(activities_path)
    if 'deal_id' not in act_df.columns:
        raise ValueError("activities.json must contain 'deal_id' column")

    act_df['timestamp'] = act_df['timestamp'].str.replace('Z$', '', regex=True)
    act_df['timestamp'] = pd.to_datetime(act_df['timestamp'], errors='coerce', utc=True)
    act_df = act_df.dropna(subset=['deal_id', 'timestamp'])

    grouped = (
        act_df
        .sort_values(['deal_id', 'timestamp'])
        .groupby('deal_id')
        .apply(lambda d: d.to_dict(orient='records'), include_groups=False)
        .reset_index(name='activities') # Correct way to rename the series values after reset_index
    )

    merged = deals_df.merge(grouped, on='deal_id', how='left')
    
    merged['activities'] = merged['activities'].apply(lambda x: x if isinstance(x, list) else [])
    
    return merged

if __name__ == "__main__":
    try:
        merged_df = load_and_merge(
            won_path="data/won_deals.json",
            lost_path="data/lost_deals.json",
            activities_path="data/activities.json"
        )
        print("Merged DataFrame head:\n", merged_df.head())
        print("Total deals loaded:", merged_df.shape[0])
        print("Columns in merged DataFrame:", merged_df.columns.tolist())
    except Exception as e:
        print(f"An error occurred: {e}")

