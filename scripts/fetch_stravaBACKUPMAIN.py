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


def strava_get_activities(access_token: str, page: int = 1, per_page: int = 200, after: int | None = None):
    url = f"https://www.strava.com/api/v3/athlete/activities?page={page}&per_page={per_page}"
    if after is not None:
        url += f"&after={after}"
    r = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Fetch activities failed: {r.status_code} {r.text}")
    return r.json()


def strava_get_activity_zones(access_token: str, activity_id: int):
    """Get power and heart rate zones for an activity"""
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/zones"
    r = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=60)
    if r.status_code == 404:
        return None  # No zones available for this activity
    if r.status_code != 200:
        print(f"[warning] Failed to get zones for activity {activity_id}: {r.status_code}")
        return None
    return r.json()


def strava_get_detailed_activity(access_token: str, activity_id: int):
    """Get detailed activity data including more fields"""
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    r = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=60)
    if r.status_code != 200:
        print(f"[warning] Failed to get detailed activity {activity_id}: {r.status_code}")
        return None
    return r.json()


def calculate_zone_based_tss(zones_data, duration_seconds, sport_type):
    """Calculate TSS based on power or HR zones"""
    if not zones_data:
        return 0
    
    tss = 0
    
    for zone_group in zones_data:
        zone_type = zone_group.get('type', '')
        
        if zone_type == 'heartrate' and sport_type in ['run', 'cycling']:
            # HR-based TSS calculation using zone distribution
            zones = zone_group.get('distribution_buckets', [])
            total_time = sum(bucket.get('time', 0) for bucket in zones)
            
            if total_time > 0:
                for i, bucket in enumerate(zones):
                    time_in_zone = bucket.get('time', 0)
                    if time_in_zone > 0:
                        # Zone multipliers (approximate IF for each HR zone)
                        # Zone 1=0.5, Zone 2=0.65, Zone 3=0.8, Zone 4=0.9, Zone 5=1.0+
                        zone_multipliers = [0.5, 0.65, 0.8, 0.9, 1.05]
                        if i < len(zone_multipliers):
                            zone_if = zone_multipliers[i]
                            zone_tss = (time_in_zone / 3600) * zone_if * zone_if * 100
                            tss += zone_tss
        
        elif zone_type == 'power' and sport_type == 'cycling':
            # Power-based TSS calculation using zone distribution
            zones = zone_group.get('distribution_buckets', [])
            
            for i, bucket in enumerate(zones):
                time_in_zone = bucket.get('time', 0)
                if time_in_zone > 0:
                    # Power zones are typically % of FTP
                    # Zone 1=<56%, Zone 2=56-75%, Zone 3=76-90%, Zone 4=91-105%, Zone 5=106-120%, Zone 6=>120%
                    zone_if_values = [0.45, 0.65, 0.83, 0.98, 1.13, 1.35]  # Approximate IF for each power zone
                    if i < len(zone_if_values):
                        zone_if = zone_if_values[i]
                        zone_tss = (time_in_zone / 3600) * zone_if * zone_if * 100
                        tss += zone_tss
    
    return round(tss)


def decode_summary_polyline(activity: dict):
    geom = activity.get("map", {}).get("summary_polyline")
    if not geom:
        return None
    # polyline.decode -> list of (lat, lon). Shapely expects (lon, lat).
    coords = polyline.decode(geom)
    if len(coords) < 2:
        return None
    lonlat = [(lon, lat) for (lat, lon) in coords]
    return LineString(lonlat)


def load_existing_geojson(path: str) -> gpd.GeoDataFrame:
    if os.path.exists(path):
        try:
            gdf = gpd.read_file(path)
            return gdf
        except Exception:
            pass
    return gpd.GeoDataFrame(columns=["id"], geometry=[], crs="EPSG:4326")


def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep a concise set of useful columns, but retain originals if present
    wanted = [
        "id",
        "name",
        "type",
        "distance",
        "moving_time",
        "elapsed_time",
        "total_elevation_gain",
        "start_date",
        "sport_type",
        "average_speed",
        "max_speed",
        "average_watts",
        "weighted_average_watts",
        "average_heartrate",
        "max_heartrate",
        "zone_based_tss",
        "kudos_count",
        "map",
    ]
    cols = [c for c in wanted if c in df.columns]
    slim = df[cols].copy()

    # Expand dates
    if "start_date" in slim.columns:
        dt = pd.to_datetime(slim["start_date"], errors="coerce", utc=True)
        slim["date"] = dt.dt.date.astype(str)
        slim["year"] = dt.dt.year
        slim["month"] = dt.dt.month

    # Convert list/dict to string for non-geometry cols (safe for shapefile)
    for c in list(slim.columns):
        if c == "map":
            continue  # we keep for geometry decoding but drop later
        if len(slim) and isinstance(slim[c].iloc[0], (list, dict)):
            slim[c] = slim[c].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

    return slim


def build_geodataframe(rows: list[dict]) -> gpd.GeoDataFrame:
    df = pd.DataFrame(rows)
    df = tidy_columns(df)
    df["geometry"] = df.apply(lambda r: decode_summary_polyline(r), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]
    # Drop the raw map column to avoid odd serialization
    if "map" in gdf.columns:
        gdf = gdf.drop(columns=["map"])
    return gdf


# -------- Main --------
if __name__ == "__main__":
    access_token, new_refresh, exp = refresh_access_token()
    if new_refresh and new_refresh != REFRESH_TOKEN:
        print("[info] Strava issued a new refresh token this run. Store it in your GitHub Secrets if you want to rotate.")

    # Find the latest recorded activity date to fetch only new ones
    existing = load_existing_geojson(GEOJSON_PATH)
    after_ts = None
    if len(existing):
        # Use max start date (ISO) to compute 'after' epoch seconds
        if "date" in existing.columns:
            try:
                last_date = pd.to_datetime(existing["date"], utc=True, errors="coerce").max()
            except Exception:
                last_date = None
        else:
            last_date = None
        if last_date is not None and pd.notnull(last_date):
            after_ts = int(last_date.timestamp())
            # Safety: subtract a day to catch late syncs
            after_ts -= 24 * 3600

    print(f"[info] Incremental fetch using after={after_ts}")

    # Pull pages
    page = 1
    new_rows = []
    while True:
        items = strava_get_activities(access_token, page=page, per_page=PER_PAGE, after=after_ts)
        if not items:
            break
        for a in items:
            new_rows.append(a)
        if len(items) < PER_PAGE:
            break
        page += 1
        time.sleep(0.5)

    print(f"[info] fetched {len(new_rows)} activities (raw) this run")

    # Enhance activities with zone-based TSS calculations
    print("[info] Calculating zone-based TSS for activities...")
    for i, activity in enumerate(new_rows):
        activity_id = activity.get('id')
        sport_type = activity.get('type', '').lower()
        
        if activity_id and sport_type in ['ride', 'run', 'cycling']:
            try:
                # Get zone distribution data
                zones_data = strava_get_activity_zones(access_token, activity_id)
                duration = activity.get('moving_time', 0)
                
                if zones_data:
                    # Calculate proper zone-based TSS
                    zone_tss = calculate_zone_based_tss(zones_data, duration, sport_type)
                    activity['zone_based_tss'] = zone_tss
                    print(f"[info] Activity {activity_id}: Zone-based TSS = {zone_tss}")
                else:
                    # Fallback to average-based calculation if no zones available
                    activity['zone_based_tss'] = 0
                    print(f"[info] Activity {activity_id}: No zone data available")
                
                # Small delay to respect API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                print(f"[warning] Failed to process zones for activity {activity_id}: {e}")
                activity['zone_based_tss'] = 0
        else:
            activity['zone_based_tss'] = 0
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"[info] Processed {i + 1}/{len(new_rows)} activities")

    # Build GeoDataFrame of new activities
    new_gdf = build_geodataframe(new_rows)

    # Merge with existing by unique id
    if len(existing):
        # Ensure 'id' is present and unique
        if "id" in existing.columns:
            combined = pd.concat([existing, new_gdf], ignore_index=True)
            combined = combined.drop_duplicates(subset=["id"], keep="last")
            combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
        else:
            combined_gdf = new_gdf
    else:
        combined_gdf = new_gdf

    # Save GeoJSON (best for web)
    if len(combined_gdf):
        combined_gdf.to_file(GEOJSON_PATH, driver="GeoJSON")
        print(f"[ok] wrote {GEOJSON_PATH}")

        # Also write ESRI Shapefile for desktop GIS
        shp_path = os.path.join(SHAPE_DIR, SHAPE_BASENAME + ".shp")
        # Shapefile column name limit: make safe
        safe = combined_gdf.copy()
        rename_map = {
            "start_date": "start_dt",
            "average_speed": "avg_speed",
            "total_elevation_gain": "elev_gain",
        }
        for k, v in rename_map.items():
            if k in safe.columns:
                safe = safe.rename(columns={k: v})
        safe.to_file(shp_path)
        print(f"[ok] wrote {shp_path} (and companion files)")

    else:
        print("[info] nothing to update")
