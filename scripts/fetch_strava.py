import os
import json
import time
import requests
import pandas as pd
import geopandas as gpd
import polyline
from shapely.geometry import LineString
from datetime import datetime

# -------- Settings --------
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")

DATA_DIR = os.path.join("docs", "data")
GEOJSON_PATH = os.path.join(DATA_DIR, "activities.geojson")
SHAPE_DIR = os.path.join(DATA_DIR, "shapefile")
SHAPE_BASENAME = "activities"
PER_PAGE = 200

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SHAPE_DIR, exist_ok=True)

# -------- Helpers --------

def refresh_access_token():
    r = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Token refresh failed: {r.status_code} {r.text}")
    tokens = r.json()
    return tokens["access_token"], tokens["refresh_token"], tokens["expires_at"]

def strava_api_error(prefix: str, r: requests.Response) -> str:
    """Human-readable message for common Strava API failures."""
    body = r.text or ""
    if r.status_code == 403 and "Inactive" in body:
        return (
            f"{prefix}: Strava API application is INACTIVE.\n"
            "Fix: open https://www.strava.com/settings/api — check your app is active.\n"
            "If needed, create a new app, re-authorize to get a fresh refresh token,\n"
            "then update GitHub secrets (STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN)."
        )
    if r.status_code == 401:
        return (
            f"{prefix}: Unauthorized (401). Refresh token may be expired or revoked.\n"
            "Re-authorize the app and update STRAVA_REFRESH_TOKEN in GitHub secrets."
        )
    return f"{prefix}: {r.status_code} {body}"


def strava_get_activities(access_token: str, page: int = 1, per_page: int = 200, after: int | None = None):
    url = f"https://www.strava.com/api/v3/athlete/activities?page={page}&per_page={per_page}"
    if after is not None:
        url += f"&after={after}"

    max_retries = 5
    base_delay = 60  # Start with 1 minute delay for rate limits

    for attempt in range(max_retries):
        r = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=60)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"[warning] Rate limit hit, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                raise RuntimeError(strava_api_error("Fetch activities failed after retries", r))
        else:
            raise RuntimeError(strava_api_error("Fetch activities failed", r))

    # This should never be reached, but just in case
    raise RuntimeError(f"Fetch activities failed after {max_retries} retries")

def strava_get_activity_zones(access_token: str, activity_id: int):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/zones"

    max_retries = 3
    base_delay = 30  # Shorter delay for zone calls

    for attempt in range(max_retries):
        r = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=60)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 404:
            return None
        elif r.status_code == 429:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[warning] Rate limit hit for zones, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                print(f"[warning] Failed to get zones for activity {activity_id} after {max_retries} retries: {r.status_code}")
                return None
        else:
            print(f"[warning] Failed to get zones for activity {activity_id}: {r.status_code}")
            return None

    return None

def strava_get_detailed_activity(access_token: str, activity_id: int):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"

    max_retries = 3
    base_delay = 30  # Shorter delay for detail calls

    for attempt in range(max_retries):
        r = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=60)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[warning] Rate limit hit for detailed activity, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                print(f"[warning] Failed to get detailed activity {activity_id} after {max_retries} retries: {r.status_code}")
                return None
        else:
            print(f"[warning] Failed to get detailed activity {activity_id}: {r.status_code}")
            return None

    return None

def calculate_zone_based_tss(zones_data, duration_seconds, sport_type):
    if not zones_data:
        return 0
    tss = 0
    for zone_group in zones_data:
        zone_type = zone_group.get('type', '')
        # Heart rate based TSS for running, cycling, swimming, and virtual rides
        if zone_type == 'heartrate' and sport_type in ['run', 'cycling', 'ride', 'virtualride', 'swim', 'swimming', 'lap_swimming', 'openwaterswim', 'openwater']:
            zones = zone_group.get('distribution_buckets', [])
            total_time = sum(bucket.get('time', 0) for bucket in zones)
            if total_time > 0:
                for i, bucket in enumerate(zones):
                    time_in_zone = bucket.get('time', 0)
                    if time_in_zone > 0:
                        zone_multipliers = [0.5, 0.65, 0.8, 0.9, 1.05]
                        if i < len(zone_multipliers):
                            zone_if = zone_multipliers[i]
                            tss += (time_in_zone / 3600) * zone_if * zone_if * 100
        # Power based TSS for cycling and virtual rides
        elif zone_type == 'power' and sport_type in ['cycling', 'ride', 'virtualride']:
            zones = zone_group.get('distribution_buckets', [])
            for i, bucket in enumerate(zones):
                time_in_zone = bucket.get('time', 0)
                if time_in_zone > 0:
                    zone_if_values = [0.45, 0.65, 0.83, 0.98, 1.13, 1.35]
                    if i < len(zone_if_values):
                        zone_if = zone_if_values[i]
                        tss += (time_in_zone / 3600) * zone_if * zone_if * 100
    return round(tss)

def decode_summary_polyline(activity: dict):
    geom = activity.get("map", {}).get("summary_polyline")
    if not geom:
        return None
    coords = polyline.decode(geom)
    if len(coords) < 2:
        return None
    lonlat = [(lon, lat) for (lat, lon) in coords]
    return LineString(lonlat)

def load_existing_geojson(path: str) -> gpd.GeoDataFrame:
    if os.path.exists(path):
        try:
            return gpd.read_file(path)
        except Exception:
            pass
    return gpd.GeoDataFrame(columns=["id"], geometry=[], crs="EPSG:4326")

def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "id","name","type","distance","moving_time","elapsed_time","total_elevation_gain",
        "start_date","sport_type","workout_type","is_race","average_speed","max_speed",
        "average_watts","weighted_average_watts","average_heartrate","max_heartrate",
        "zone_based_tss","kudos_count","map"
    ]
    cols = [c for c in wanted if c in df.columns]
    slim = df[cols].copy()
    if "start_date" in slim.columns:
        dt = pd.to_datetime(slim["start_date"], errors="coerce", utc=True)
        slim["date"] = dt.dt.date.astype(str)
        slim["year"] = dt.dt.year
        slim["month"] = dt.dt.month
    for c in list(slim.columns):
        if c == "map":
            continue
        if len(slim) and isinstance(slim[c].iloc[0], (list, dict)):
            slim[c] = slim[c].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    return slim

def build_geodataframe(rows: list[dict]) -> gpd.GeoDataFrame:
    df = pd.DataFrame(rows)
    df = tidy_columns(df)
    df["geometry"] = df.apply(lambda r: decode_summary_polyline(r), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    # Keep all activities, including those without geometry (indoor activities)
    # Only filter out invalid geometries if they exist
    gdf = gdf[gdf.geometry.isnull() | gdf.geometry.is_valid]
    if "map" in gdf.columns:
        gdf = gdf.drop(columns=["map"])
    return gdf

# -------- Main --------
if __name__ == "__main__":
    access_token, new_refresh, exp = refresh_access_token()
    if new_refresh and new_refresh != REFRESH_TOKEN:
        print("[info] Strava issued a new refresh token this run. Store it in your GitHub Secrets if you want to rotate.")

    # FULL FETCH MODE: Fetch all activities from the beginning
    print("[info] Full fetch mode: fetching ALL activities from Strava (this may take time)")

    # ---------------- Fetch all activities ----------------
    page = 1
    new_rows = []
    while True:
        items = strava_get_activities(access_token, page=page, per_page=PER_PAGE, after=None)
        if not items:
            break
        new_rows.extend(items)
        if len(items) < PER_PAGE:
            break
        page += 1
        time.sleep(0.5)

    print(f"[info] fetched {len(new_rows)} activities (raw) this run")
    
    # Debug: Show activity types fetched
    if new_rows:
        activity_types = {}
        for activity in new_rows:
            sport_type = activity.get("type", "").lower()
            activity_types[sport_type] = activity_types.get(sport_type, 0) + 1
        print(f"[info] Activity types fetched: {dict(sorted(activity_types.items()))}")

    # ---------------- Enrich recent activities ----------------
    DAYS_TO_ENRICH = 30
    cutoff_ts = int(datetime.utcnow().timestamp()) - DAYS_TO_ENRICH * 24*3600

    for i, activity in enumerate(new_rows):
        activity_id = activity.get("id")
        sport_type = activity.get("type", "").lower()
        start_ts = int(pd.to_datetime(activity.get("start_date"), utc=True).timestamp())

        if start_ts >= cutoff_ts:
            if 'workout_type' not in activity or activity.get('workout_type') is None:
                detailed = strava_get_detailed_activity(access_token, activity_id)
                if detailed and 'workout_type' in detailed:
                    activity['workout_type'] = detailed.get('workout_type')

            wt = activity.get('workout_type')
            activity['is_race'] = bool(wt == 1)

            # Include indoor cycling (virtualride) and swimming activities
            # Swimming types: swim, swimming, lap_swimming, openwaterswim, openwater
            if sport_type in ['ride', 'run', 'cycling', 'virtualride', 'swim', 'swimming', 'lap_swimming', 'openwaterswim', 'openwater']:
                zones_data = strava_get_activity_zones(access_token, activity_id)
                duration = activity.get('moving_time', 0)
                if zones_data:
                    activity['zone_based_tss'] = calculate_zone_based_tss(zones_data, duration, sport_type)
                else:
                    activity['zone_based_tss'] = 0

                time.sleep(0.2)  # short sleep for zones/detail calls
        else:
            activity['is_race'] = bool(activity.get('workout_type') == 1)
            activity['zone_based_tss'] = 0

        if (i+1) % 50 == 0:
            print(f"[info] Processed {i+1}/{len(new_rows)} activities")

    # ---------------- Build GeoDataFrame ----------------
    new_gdf = build_geodataframe(new_rows)
    print(f"[info] Built GeoDataFrame with {len(new_gdf)} activities (after filtering)")

    # Since we're doing a full fetch, use only the newly fetched data
    # Remove any duplicates within the fetched data (shouldn't happen but just in case)
    new_gdf = new_gdf.drop_duplicates(subset=["id"], keep="last")
    combined_gdf = new_gdf

    # Save GeoJSON
    if len(combined_gdf):
        combined_gdf.to_file(GEOJSON_PATH, driver="GeoJSON")
        print(f"[ok] wrote {GEOJSON_PATH}")

        # Save Shapefile
        shp_path = os.path.join(SHAPE_DIR, SHAPE_BASENAME + ".shp")
        safe = combined_gdf.copy()
        rename_map = {
            "start_date": "start_dt",
            "average_speed": "avg_speed",
            "total_elevation_gain": "elev_gain",
        }
        for k,v in rename_map.items():
            if k in safe.columns:
                safe = safe.rename(columns={k:v})
        safe.to_file(shp_path)
        print(f"[ok] wrote {shp_path} (and companion files)")
    else:
        print("[info] nothing to update")

