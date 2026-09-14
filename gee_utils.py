"""
gee_utils.py — Google Earth Engine & STAC Fallback Utilities for EPSIS
=======================================================================
Provides satellite image acquisition helpers with explicit bounding box
clipping to ensure spatial registration between T1 and T2 images.
"""

import os
import tempfile
import requests
import math
from config import GCP_PROJECT_ID, CLOUD_FILTER_MAX_PERCENT

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

GEE_PROJECT = GCP_PROJECT_ID
CLOUD_FILTER_PERCENT = CLOUD_FILTER_MAX_PERCENT
RGB_BANDS = ["B4", "B3", "B2"]
VIS_MIN, VIS_MAX = 0, 3000
THUMBNAIL_DIMENSIONS = 512

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


def get_satellite_image(latitude, longitude, start_date, end_date, buffer_m=1000):
    """
    Fetches least cloudy Sentinel-2 image for target location and date range.
    Returns (ee_image, region_bbox_geometry).
    """
    if not _GEE_INITIALIZED:
        raise ValueError("Google Earth Engine is not initialized or authenticated.")

    bbox = get_bounding_box(latitude, longitude, buffer_m)
    region = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(str(start_date), str(end_date))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER_PERCENT))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(
            f"No Sentinel-2 images found between {start_date} and {end_date} "
            f"with <{CLOUD_FILTER_PERCENT}% cloud cover at ({latitude:.4f}, {longitude:.4f})."
        )

    return collection.first(), region


def get_image_thumbnail(image, region=None) -> str:
    """
    Returns thumbnail URL for ee.Image with explicit region clipping.
    """
    if image is None:
        raise ValueError("get_image_thumbnail() received None.")

    vis_params = {
        "bands": RGB_BANDS,
        "min": VIS_MIN,
        "max": VIS_MAX,
        "gamma": 1.2,
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
