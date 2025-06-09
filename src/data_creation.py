import json
import datetime
import random
import uuid
import os

def generate_activities_json(won_deals_file, lost_deals_file, output_filename='activities.json'):
    """
    Generates a synthetic activities.json file based on deal data.

    Args:
        won_deals_file (str): Path to the won deals JSON file.
        lost_deals_file (str): Path to the lost deals JSON file.
        output_filename (str): Name of the output activities JSON file.
    """
    try:
        # Load won_deals_train.json
        with open(won_deals_file, 'r') as f:
            won_deals = json.load(f)
        print(f"Loaded {len(won_deals)} deals from {won_deals_file}")

        # Load lost_deals_train.json
        with open(lost_deals_file, 'r') as f:
            lost_deals = json.load(f)
        print(f"Loaded {len(lost_deals)} deals from {lost_deals_file}")

    except FileNotFoundError as e:
        print(f"Error: One of the deal files not found. Please ensure '{won_deals_file}' and '{lost_deals_file}' are in the same directory as the script. Error: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from deal files: {e}")
        return

    all_deals = won_deals + lost_deals
    activities = []
    activity_types = ['Call', 'Email', 'Meeting', 'Note', 'Task']

    # Collect all unique owner IDs from the deals
    owner_ids = list(set([
        deal['properties']['hubspot_owner_id']
        for deal in all_deals
        if 'hubspot_owner_id' in deal['properties']
    ]))

    if not owner_ids:
        print("Warning: No 'hubspot_owner_id' found in any deals. Activities will not have an owner_id.")

    for deal in all_deals:
        deal_id = deal.get('id')
        if not deal_id:
            print(f"Skipping deal with no ID: {deal}")
            continue

        deal_properties = deal.get('properties', {})
        deal_createdate_str = deal_properties.get('createdate')
        deal_closedate_str = deal_properties.get('closedate')
        deal_owner_id = deal_properties.get('hubspot_owner_id')

        if not deal_createdate_str:
            # print(f"Skipping deal {deal_id} due to missing 'createdate'.")
            continue # Skip deals without a creation date

        try:
            created_at = datetime.datetime.fromisoformat(deal_createdate_str.replace('Z', '+00:00'))
        except ValueError:
            print(f"Warning: Invalid 'createdate' format for deal {deal_id}: {deal_createdate_str}. Skipping activities for this deal.")
            continue

        # Determine activity generation range
        end_date = datetime.datetime.now(datetime.timezone.utc) # Default end date is now

        if deal_closedate_str:
            try:
                closed_at = datetime.datetime.fromisoformat(deal_closedate_str.replace('Z', '+00:00'))
                # If closedate is in the future, use now as the end date for activities
                if closed_at < end_date: # Only use closed_at if it's in the past
                    end_date = closed_at
            except ValueError:
                print(f"Warning: Invalid 'closedate' format for deal {deal_id}: {deal_closedate_str}. Using current date as end date.")
                # Fallback to current date if closedate is invalid

        # Ensure start date is not after end date
        # This handles cases where created_at might be in the future, or end_date becomes earlier
        start_date = min(created_at, end_date)
        end_date = max(created_at, end_date)

        # Ensure there's a valid time difference for activity generation
        if (end_date - start_date).total_seconds() <= 0:
            # print(f"Warning: No valid date range for deal {deal_id}. Skipping activities.")
            continue

        # Generate a random number of activities for each deal (e.g., 1 to 5)
        num_activities = random.randint(1, 5)

        for _ in range(num_activities):
            activity_id = str(uuid.uuid4())
            activity_type = random.choice(activity_types)

            # Generate a random timestamp between start_date and end_date
            time_diff_seconds = (end_date - start_date).total_seconds()
            if time_diff_seconds > 0:
                random_seconds = random.uniform(0, time_diff_seconds)
                activity_timestamp = start_date + datetime.timedelta(seconds=random_seconds)
            else:
                activity_timestamp = start_date # If no time diff, just use start_date

            # Format timestamp to ISO 8601 with 'Z' for UTC
            formatted_timestamp = activity_timestamp.isoformat(timespec='milliseconds') + 'Z'

            # Pick an owner, prefer deal owner if available, otherwise a random one from collected IDs
            chosen_owner = deal_owner_id
            if not chosen_owner and owner_ids:
                chosen_owner = random.choice(owner_ids)

            activity_description = f"{activity_type} related to deal {deal_id}"

            activity_entry = {
                'activity_id': activity_id,
                'deal_id': deal_id,
                'activity_type': activity_type,
                'timestamp': formatted_timestamp,
                'description': activity_description
            }
            if chosen_owner:
                activity_entry['owner_id'] = chosen_owner

            activities.append(activity_entry)

    # Save to activities.json
    with open(output_filename, 'w') as f:
        json.dump(activities, f, indent=4)

    print(f"\nSuccessfully generated {len(activities)} activities and saved to '{output_filename}'")
    print("You can now use this file in your analysis.")

if __name__ == "__main__":
    # Ensure these paths are correct relative to where you run the script
    # Or provide full paths to your won_deals_train.json and lost_deals_train.json
    generate_activities_json('data\won_deals_train.json', 'data\lost_deals_train.json', 'activities.json')
