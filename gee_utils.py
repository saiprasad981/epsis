import os
import tempfile

import ee
import requests

GEE_PROJECT = "epsis-502113"  # <-- set this to your actual GCP project id

ee.Initialize(project=GEE_PROJECT)

CLOUD_FILTER_PERCENT = 10
RGB_BANDS = ["B4", "B3", "B2"]
VIS_MIN, VIS_MAX = 0, 3000
THUMBNAIL_DIMENSIONS = 1024  # display quality; inference.py resizes to 256 separately


def get_satellite_image(latitude, longitude, start_date, end_date):
    """
    Fetches the least cloudy Sentinel-2 SR Harmonized image for the given
    location and date range. Returns an ee.Image (a single image, not a
    collection) — raises a clear ValueError if nothing matches.
    """
    point = ee.Geometry.Point([longitude, latitude])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(str(start_date), str(end_date))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER_PERCENT))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(
            f"No Sentinel-2 images found between {start_date} and {end_date} "
            f"with <{CLOUD_FILTER_PERCENT}% cloud cover at ({latitude}, {longitude}). "
            f"Try widening the date range."
        )

    return collection.first()


def get_image_thumbnail(image) -> str:
    """
    Returns a thumbnail URL (string) for displaying a single ee.Image in
    Streamlit via st.image(). This is a URL, not local image bytes — use
    save_temp_image() below to get an actual file on disk for model input.
    """
    if image is None:
        raise ValueError("get_image_thumbnail() received None instead of an ee.Image.")

    vis_params = {
        "bands": RGB_BANDS,
        "min": VIS_MIN,
        "max": VIS_MAX,
        "gamma": 1.2,
        "dimensions": THUMBNAIL_DIMENSIONS,
        "format": "png",
    }
    return image.getThumbURL(vis_params)


def save_temp_image(thumbnail_url: str) -> str:
    """
    Downloads a GEE thumbnail URL and saves it as a local PNG file, since
    inference.py's predict_change() needs an actual file path (PIL.Image.open),
    not a URL. Returns the local file path.
    """
    response = requests.get(thumbnail_url, timeout=30)
    response.raise_for_status()

    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_file.write(response.content)
    tmp_file.close()

    print(f"[gee_utils] Saved temp image: {tmp_file.name} ({len(response.content)} bytes)")
    return tmp_file.name
