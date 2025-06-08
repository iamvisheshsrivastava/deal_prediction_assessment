# src/feature_engineering.py

import pandas as pd
import numpy as np

def count_activity_types(activities: list) -> dict:
    """
    Given a list of activity dicts (each with 'type' and 'timestamp'),
    return counts per activity type.
    """
    types = ["Email", "Meeting", "Task", "Note", "Call"]
    counts = dict.fromkeys(types, 0)
    for act in activities:
        t = act.get('type')
        if t in counts:
            counts[t] += 1
    return counts

def extract_temporal_features(activities: list) -> dict:
    """
    Derive features such as:
      - duration (days) between first and last activity
      - average time delta between activities (in hours)
      - activity rate per day
    """
    if not activities:
        return {
            "duration_days": 0,
            "avg_delta_hours": 0,
            "activity_rate_per_day": 0
        }
    # Extract and sort timestamps
    timestamps = [pd.to_datetime(act['timestamp']) for act in activities]
    timestamps = sorted(timestamps)
    # Duration
    duration = (timestamps[-1] - timestamps[0]).total_seconds() / (3600*24)  # in days
    # Compute deltas between consecutive activities
    deltas = np.diff([ts.value for ts in timestamps])  # in nanoseconds
    # Convert to hours if at least 2 activities
    if len(deltas) > 0:
        avg_delta_hr = np.mean(deltas) / 1e9 / 3600
    else:
        avg_delta_hr = 0
    # Activity rate: total activities / (duration_days + 1e-6)
    rate = len(timestamps) / (duration + 1e-6)
    return {
        "duration_days": duration,
        "avg_delta_hours": avg_delta_hr,
        "activity_rate_per_day": rate
    }

def extract_unstructured_features(activities: list) -> dict:
    """
    Example: For each 'Note' activity, count total words.
    You can expand to sentiment analysis, keyword counts, etc.
    """
    total_note_words = 0
    for act in activities:
        if act.get('type') == "Note":
            text = act.get('content', "")
            total_note_words += len(text.split())
    return {"total_note_words": total_note_words}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns 'deal_id', 'outcome', and 'activities' (list),
    return a new DataFrame of features + target.
    """
    records = []
    for _, row in df.iterrows():
        deal_id = row['deal_id']
        activities = row['activities']
        label = row['outcome']
        
        # Quantitative
        counts = count_activity_types(activities)
        # Temporal
        temporal = extract_temporal_features(activities)
        # Unstructured
        unstructured = extract_unstructured_features(activities)
        
        features = {
            "deal_id": deal_id,
            "outcome": label,
            **counts,
            **temporal,
            **unstructured
        }
        records.append(features)
    feat_df = pd.DataFrame(records).drop(columns=['deal_id'])
    # If any NaNs, fill with 0
    feat_df = feat_df.fillna(0)
    return feat_df

if __name__ == "__main__":
    # Quick test: merge first, then build features
    from data_processing import load_and_merge
    merged_df = load_and_merge("data/won_deals.json",
                               "data/lost_deals.json",
                               "data/activities.json")
    features_df = build_features(merged_df)
    print("Features DataFrame head:\n", features_df.head())
    print("Feature columns:", features_df.columns.tolist())
