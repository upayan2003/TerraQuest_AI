import ee
import geemap
import numpy as np
from datetime import datetime, timedelta
import cv2
import os

# Authenticate with Earth Engine using service account
def authenticate_earth_engine():
    """
    Authenticate with Earth Engine using a service account.
    Ensure you have the service account JSON key file.
    """
    SERVICE_ACCOUNT = 'gee-webapp@ee-upayan2003.iam.gserviceaccount.com'
    KEY_FILE = 'ee-upayan2003-3fb46ffa3e11.json'

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

    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                  .filterBounds(region) \
                  .filterDate(start_date, end_date) \
                  .map(mask_s2_clouds)

    image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first().select(bands).clip(region)

    # collection = (
    #     ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    #     .filterBounds(region)
    #     .filterDate('2025-01-01', '2025-06-01')
    #     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    #     .map(mask_s2_clouds)
    #     .sort('system:time_start', False)  # most recent first
    # )

    # image = collection.first().select(bands).clip(region)

    if image is None:
        raise ValueError("No suitable image found for the specified location and date range.")

    vis_image = image.visualize(min=0, max=3000, bands=bands)

    # Export using geemap
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

        return img_resized

    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")


# Example usage
if __name__ == "__main__":
    authenticate_earth_engine()
    
    # Get coordinates from user input
    coords = input("Enter coordinates (lat, lon): ")
    lat, lon = map(float, coords.split(','))

    try:
        img_array = get_satellite_patch(lat, lon, output_path='patch.png')
        print("Patch shape:", img_array.shape)  # Should be (256, 256, 3)
    except Exception as e:
        print(f"Error fetching patch: {e}. Input valid land coordinates.")
