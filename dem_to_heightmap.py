import subprocess
import numpy as np
from PIL import Image
from osgeo import gdal

INPUT_TIF    = "/home/mchalak/Desktop/map/rasters_COP30/output_hh.tif"
OUTPUT_PNG   = "/home/mchalak/marid_ws/src/marid_description/worlds/terrain_world/materials/textures/heightmap.png"
WORLD_SIZE_M = 4000   # real-world size in meters (4 km x 4 km)
IMG_SIZE     = 513    # must be 2^n + 1: 129, 257, 513, 1025

reprojected = "/tmp/dem_reproj.tif"

subprocess.run([
    "gdalwarp",
    "-t_srs", "EPSG:32610",
    "-ts", str(IMG_SIZE), str(IMG_SIZE),
    "-r", "bilinear",
    INPUT_TIF, reprojected
], check=True)

ds   = gdal.Open(reprojected)
band = ds.GetRasterBand(1)
arr  = band.ReadAsArray().astype(np.float32)

# Handle ALOS/Copernicus voids
arr[arr <= -9000] = np.nan
arr = np.where(np.isnan(arr), np.nanmin(arr), arr)

z_min, z_max = arr.min(), arr.max()
print(f"Elevation range: {z_min:.1f} m  to  {z_max:.1f} m")
print(f"Height span:     {z_max - z_min:.1f} m")

arr_norm = ((arr - z_min) / (z_max - z_min) * 65535).astype(np.uint16)

import os
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
Image.fromarray(arr_norm, mode='I;16').save(OUTPUT_PNG)
print(f"Saved: {OUTPUT_PNG}")
print(f"\nUse in SDF:  <size>{WORLD_SIZE_M} {WORLD_SIZE_M} {z_max - z_min:.1f}</size>")
