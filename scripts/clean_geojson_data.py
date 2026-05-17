import json
import os
from datetime import datetime

DATA_DIR = os.path.join("docs", "data")
ACTIVITIES_PATH = os.path.join(DATA_DIR, "activities.geojson")
TRAVEL_PATH = os.path.join(DATA_DIR, "travel.geojson")

# Cutoff date: December 11, 2025 (keep activities BEFORE this date, so Dec 10 is included)
CUTOFF_DATE = datetime(2025, 12, 11)

def clean_geojson_file(file_path, cutoff_date):
    """Remove all features that occurred on or after the cutoff date."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist, skipping...")
        return

    print(f"Processing {file_path}...")

    with open(file_path, 'r') as f:
        data = json.load(f)

    original_count = len(data['features'])
    print(f"Original feature count: {original_count}")

    # Filter features based on date
    filtered_features = []
    removed_count = 0

    for feature in data['features']:
        date_str = feature['properties'].get('date')
        if date_str:
            try:
                # Parse date (format: "2025-12-10 00:00:00")
                activity_date = datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')

                if activity_date >= cutoff_date:
                    removed_count += 1
                    continue  # Skip this feature
            except ValueError:
                print(f"Warning: Could not parse date '{date_str}', keeping feature")
                pass

        filtered_features.append(feature)

    new_count = len(filtered_features)
    print(f"Removed {removed_count} features (kept {new_count})")

    # Save the filtered data
    data['features'] = filtered_features

    # Create backup
    backup_path = file_path + '.backup'
    if not os.path.exists(backup_path):
        print(f"Creating backup: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)

    # Save cleaned data
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved cleaned data to {file_path}")
    print(f"Backup created at {backup_path}")
    print()

if __name__ == "__main__":
    print(f"Cleaning data after {CUTOFF_DATE.strftime('%Y-%m-%d')}...")
    print("=" * 50)

    clean_geojson_file(ACTIVITIES_PATH, CUTOFF_DATE)
    clean_geojson_file(TRAVEL_PATH, CUTOFF_DATE)

    print("Data cleaning complete!")
    print("You can now re-run the fetch scripts to get fresh data.")