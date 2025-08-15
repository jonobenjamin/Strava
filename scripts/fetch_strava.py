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
