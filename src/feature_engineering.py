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
        t = act.get('activity_type') 
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
    timestamps = [pd.to_datetime(act['timestamp']) for act in activities if 'timestamp' in act]
    
    if not timestamps: 
        return {
            "duration_days": 0,
            "avg_delta_hours": 0,
            "activity_rate_per_day": 0
        }

    timestamps = sorted(timestamps)
    
    if len(timestamps) > 1:
        duration = (timestamps[-1] - timestamps[0]).total_seconds() / (3600*24)  
    else:
        duration = 0 

    deltas = np.diff([ts.value for ts in timestamps]) 
    
    if len(deltas) > 0:
        avg_delta_hr = np.mean(deltas) / 1e9 / 3600
    else:
        avg_delta_hr = 0
    
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
        if act.get('activity_type') == "Note": 
            text = act.get('description', "") 
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
        
        counts = count_activity_types(activities)
        temporal = extract_temporal_features(activities)
        unstructured = extract_unstructured_features(activities)
        
        features = {
            "deal_id": deal_id,
            "outcome": label,
            **counts,
            **temporal,
            **unstructured
        }
        records.append(features)
    
    feat_df = pd.DataFrame(records)
    
    if 'deal_id' in feat_df.columns:
        feat_df = feat_df.drop(columns=['deal_id'])
    
    feat_df = feat_df.fillna(0)
    return feat_df

if __name__ == "__main__":
    try:
        from src.data_processing import load_and_merge 
        
        merged_df = load_and_merge("data/won_deals.json",
                                   "data/lost_deals.json",
                                   "data/activities.json")
        features_df = build_features(merged_df)
        print("Features DataFrame head:\n", features_df.head())
        print("Feature columns:", features_df.columns.tolist())
        print("Features DataFrame info:\n", features_df.info()) # Added for more detailed check
    except FileNotFoundError as e:
        print(f"Error: Ensure data files are in 'data/' directory. {e}")
    except Exception as e:
        print(f"An error occurred during feature engineering: {e}")

