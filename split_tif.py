import cv2 as cv
import numpy as np
import os
import rasterio as rio
import sys

# Open the raster file

IMG_PATH = sys.argv[1]
TILE_SIZE = int(sys.argv[2])
OUT_DIR = 'figures/kenya'
os.makedirs(OUT_DIR, exist_ok=True)


with rio.open(IMG_PATH) as src:
	widht, height = src.width, src.height
	num_tiles_x = widht // TILE_SIZE
	num_tiles_y = height // TILE_SIZE

	for i in range(num_tiles_x):
		for j in range(num_tiles_y):
			window = rio.windows.Window(i*TILE_SIZE, j*TILE_SIZE, TILE_SIZE, TILE_SIZE)
			clip = src.read(window=window)

			clip = np.moveaxis(clip, 0, -1)
			clip = cv.cvtColor(clip, cv.COLOR_RGBA2BGR)

			out_path = os.path.join(OUT_DIR, f"tile_{i}_{j}.tif")
			cv.imwrite(out_path, clip)


