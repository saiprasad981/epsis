"""
Test script to compare GEE nearest neighbor vs bicubic resampling vs native 10m scale
"""
import ee
import datetime
import requests
import io
from PIL import Image
import config

ee.Initialize(project=config.GCP_PROJECT_ID)

# Hyderabad Hitech City / Kokapet
lat, lon = 17.3950, 78.3300
buffer_m = 1000.0  # 1km radius -> 2km x 2km bounding box

lat_delta = buffer_m / 111320.0
lon_delta = buffer_m / (111320.0 * 0.9542)

bbox = [lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta]
region = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])

# Query scene for 2021-06-15
col = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate("2021-06-01", "2021-06-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

features = col.limit(5).getInfo().get("features", [])
best_id = features[0]["id"]
print("Selected Scene ID:", best_id)

raw_img = ee.Image(best_id).clip(region)
bicubic_img = ee.Image(best_id).resample("bicubic").clip(region)

vis_params = {
    "bands": ["B4", "B3", "B2"],
    "min": 0,
    "max": 3000,
    "gamma": 1.1,
    "dimensions": 1024,
    "region": region,
    "format": "png",
}

# 1. Default GEE nearest neighbor
url_raw = raw_img.getThumbURL(vis_params)
r_raw = requests.get(url_raw)
img_raw = Image.open(io.BytesIO(r_raw.content))
img_raw.save("test_nearest.png")
print("Saved test_nearest.png (Nearest neighbor 1024x1024):", img_raw.size)

# 2. Bicubic resampling
url_bicubic = bicubic_img.getThumbURL(vis_params)
r_bicubic = requests.get(url_bicubic)
img_bicubic = Image.open(io.BytesIO(r_bicubic.content))
img_bicubic.save("test_bicubic.png")
print("Saved test_bicubic.png (Bicubic resampled 1024x1024):", img_bicubic.size)

# 3. Native 10m scale output (no forced 1024 dimension)
vis_native = {
    "bands": ["B4", "B3", "B2"],
    "min": 0,
    "max": 3000,
    "gamma": 1.1,
    "scale": 10,
    "region": region,
    "format": "png",
}
url_native = raw_img.getThumbURL(vis_native)
r_native = requests.get(url_native)
img_native = Image.open(io.BytesIO(r_native.content))
img_native.save("test_native.png")
print("Saved test_native.png (Native 10m scale):", img_native.size)
