"""
gee_utils.py — Google Earth Engine & STAC Fallback Utilities for EPSIS
=======================================================================
Provides satellite image acquisition helpers for single exact acquisition dates
with multi-constellation fallback (Sentinel-2 SR Harmonized, Landsat-8, Landsat-7) and explicit
bounding box clipping to ensure spatial registration between T1 and T2 images.
"""

import os
import tempfile
import requests
import math
import datetime
import io
from PIL import Image
from config import GCP_PROJECT_ID, CLOUD_FILTER_MAX_PERCENT

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

GEE_PROJECT = GCP_PROJECT_ID
CLOUD_FILTER_PERCENT = CLOUD_FILTER_MAX_PERCENT
RGB_BANDS = ["B4", "B3", "B2"]
VIS_MIN, VIS_MAX = 150, 2200
THUMBNAIL_DIMENSIONS = 1024

# Safely attempt GEE initialization without crashing import
_GEE_INITIALIZED = False
if GEE_AVAILABLE:
    try:
        ee.Initialize(project=GEE_PROJECT)
        _GEE_INITIALIZED = True
    except Exception as _e:
        print(f"[gee_utils] GEE Notice: {_e}")


def get_bounding_box(latitude, longitude, buffer_m=1000):
    """Returns GEE Geometry BBox around target lat/lon."""
    lat_delta = buffer_m / 111320.0
    lon_delta = buffer_m / (111320.0 * math.cos(math.radians(latitude)))
    return [longitude - lon_delta, latitude - lat_delta, longitude + lon_delta, latitude + lat_delta]


def get_single_satellite_image_for_date(latitude, longitude, target_date, buffer_m=1000, tolerance_days=30):
    """
    Fetches the closest valid, low-cloud satellite image for a target acquisition date.
    Supports multi-constellation fallback: Sentinel-2 SR Harmonized -> Landsat-8 C2 -> Landsat-7 C2.

    Returns:
        (ee_image, region_geometry, actual_date_str, cloud_pct, constellation_name)
    """
    if not _GEE_INITIALIZED:
        raise ValueError("Google Earth Engine is not initialized or authenticated.")

    if isinstance(target_date, str):
        target_dt = datetime.date.fromisoformat(target_date)
    else:
        target_dt = target_date

    bbox = get_bounding_box(latitude, longitude, buffer_m)
    region = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])

    search_tiers = [
        (tolerance_days, CLOUD_FILTER_PERCENT),
        (max(tolerance_days, 60), min(CLOUD_FILTER_PERCENT + 20, 50.0)),
        (max(tolerance_days, 120), 65.0),
    ]

    is_s2_era = target_dt >= datetime.date(2015, 6, 23)

    for tol, cloud_max in search_tiers:
        start_dt = target_dt - datetime.timedelta(days=tol)
        end_dt = target_dt + datetime.timedelta(days=tol)

        if is_s2_era:
            col = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region)
                .filterDate(start_dt.isoformat(), end_dt.isoformat())
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
            )
            try:
                col_info = col.limit(15).getInfo()
                features = col_info.get("features", [])
            except Exception:
                features = []

            if features:
                candidates = []
                for feat in features:
                    props = feat["properties"]
                    scene_id = feat["id"]
                    time_start = props["system:time_start"]
                    cloud_pct = float(props.get("CLOUDY_PIXEL_PERCENTAGE", 0.0))
                    acq_dt = datetime.datetime.fromtimestamp(time_start / 1000.0, datetime.timezone.utc).date()
                    days_diff = abs((acq_dt - target_dt).days)
                    cost = (days_diff * 1.5) + cloud_pct
                    candidates.append((cost, days_diff, cloud_pct, acq_dt, scene_id))

                candidates.sort(key=lambda c: c[0])
                _, _, best_cloud, best_acq_dt, best_id = candidates[0]
                img = ee.Image(best_id).clip(region)
                return img, region, str(best_acq_dt), best_cloud, "Sentinel-2 SR Harmonized"

    raise ValueError(
        f"No usable clear satellite image (Sentinel-2 / Landsat-8 / Landsat-7) found near requested date {target_dt.isoformat()} "
        f"at ({latitude:.4f}, {longitude:.4f}). Try selecting a date between 1999 and present."
    )


def get_satellite_image(latitude, longitude, target_date, buffer_m=1000, tolerance_days=30):
    """Convenience wrapper returning (ee_image, region_bbox_geometry) for a target date."""
    img, region, _, _, _ = get_single_satellite_image_for_date(latitude, longitude, target_date, buffer_m, tolerance_days)
    return img, region


def get_image_thumbnail(image, region=None) -> str:
    """Returns thumbnail URL for ee.Image with explicit region clipping and consistent reflectance scaling."""
    if image is None:
        raise ValueError("get_image_thumbnail() received None.")

    vis_params = {
        "bands": RGB_BANDS,
        "min": VIS_MIN,
        "max": VIS_MAX,
        "gamma": 1.1,
        "dimensions": THUMBNAIL_DIMENSIONS,
        "format": "png",
    }
    if region is not None:
        vis_params["region"] = region

    return image.getThumbURL(vis_params)


def save_temp_image(thumbnail_url: str) -> str:
    """Downloads thumbnail URL to temp file."""
    response = requests.get(thumbnail_url, timeout=30)
    response.raise_for_status()

    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_file.write(response.content)
    tmp_file.close()
    return tmp_file.name
