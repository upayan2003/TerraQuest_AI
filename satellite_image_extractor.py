import ee
import json
import tempfile
import geemap
import numpy as np
from datetime import datetime, timedelta
import cv2
import os

# Authenticate with Earth Engine using service account
def authenticate_earth_engine(gcloud_json_str):
    """
    Authenticate with Earth Engine using a service account JSON string.
    """
    # Write the JSON string to a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmpfile:
        tmpfile.write(gcloud_json_str)
        tmpfile.flush()

        SERVICE_ACCOUNT = json.loads(gcloud_json_str)["client_email"]
        KEY_FILE = tmpfile.name

        credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)

        ee.Initialize(credentials)
        print("Authenticated and Earth Engine initialized.")

# Function to mask clouds in Sentinel-2 images

def mask_s2_clouds(image):
    """ Masks clouds in Sentinel-2 images using the QA60 band."""

    # Get the QA60 band (quality band for clouds and cirrus)
    qa = image.select("QA60")
    
    # Bits 10 and 11 are clouds and cirrus respectively
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    
    # Both bits should be 0 (no cloud, no cirrus)
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])

# Function to fetch a satellite image patch

def get_satellite_patch(lat, lon, size_px=256, scale=10, bands=['B4', 'B3', 'B2'], output_path=None):
    """
    Fetches a 256x256 satellite image patch centered on the lat/lon.
    Also returns mean NDBI value of the patch.

    Returns:
        - np.ndarray image (H x W x C)
        - Saves image as PNG if output_path is specified
    """
    buffer_m = size_px * scale / 2  # Half-size buffer
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m).bounds()

    today = datetime.today()
    four_months_ago = (today.replace(day=1) - timedelta(days=120)).replace(day=1)

    # Format as strings for Earth Engine
    start_date = four_months_ago.strftime("%Y-%m-%d")
    end_date = today.replace(day=1).strftime("%Y-%m-%d")

    # Choose a Sentinel-2 SR image with low cloud cover
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(region)
                .filterDate(start_date, end_date)
                .map(mask_s2_clouds))

    # Fetch image with required bands
    image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first().select(['B2', 'B3', 'B4']).clip(region)

    if image is None:
        raise ValueError("No suitable image found for the specified location and date range.")

    # Load VIIRS Nighttime Lights data
    nightlight_img = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
                    .filterBounds(region)
                    .filterDate(start_date, end_date)
                    .select("avg_rad")
                    .mean()  # mean over the month(s)
                    .clip(region))

    # Compute mean nightlight radiance in region
    nightlight_stat = nightlight_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=500,  # VIIRS resolution
        maxPixels=1e9
    )

    # Extract radiance value
    nightlight = nightlight_stat.get('avg_rad').getInfo() if nightlight_stat.get('avg_rad') else 0

    # Visualize RGB image
    vis_image = image.select(['B4', 'B3', 'B2']).visualize(min=0, max=3000)

    temp_file = 'patch.tif'
    geemap.ee_export_image(
        vis_image,
        filename=temp_file,
        scale=scale,
        region=region,
        file_per_band=False
    )

    try:
        img = cv2.imread(temp_file, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise RuntimeError("Image file could not be loaded.")

        img_resized = cv2.resize(img, (size_px, size_px), interpolation=cv2.INTER_CUBIC)

        if output_path:
            cv2.imwrite(output_path, img_resized)

        os.remove(temp_file)

        return img_resized, nightlight

    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

# Example usage
if __name__ == "__main__":
    authenticate_earth_engine()
    
    # Get coordinates from user input
    coords = input("Enter coordinates (lat, lon): ")
    lat, lon = map(float, coords.split(','))

    try:
        img_array, nlight = get_satellite_patch(lat, lon, output_path='patch.png')
        print("Patch shape:", img_array.shape)  # Should be (256, 256, 3)
        print("Nightlight Radiance:", nlight)
    except Exception as e:
        print(f"Error fetching patch: {e}. Input valid land coordinates.")