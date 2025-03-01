import rasterio as rio
from rasterio.windows import Window
import os
from test_hls import detect_pumpkins_hls
import cv2 as cv
import numpy as np

def split_geotiff(input_path, output_folder, tile_size=512):
	os.makedirs(output_folder, exist_ok=True)
	total_count = 0

	with rio.open(input_path) as dataset:
		img_width, img_height = dataset.width, dataset.height

		total_ocunt = 0

		num_tiles_x = img_width // tile_size
		num_tiles_y = img_height // tile_size

		idx = 0
		# Iterate through the image in tile_size steps
		for i in range(num_tiles_x):
			for j in range(num_tiles_y):
				window = Window(i*tile_size, j*tile_size, tile_size, tile_size)
				tile = dataset.read(window=window)
				tile = np.moveaxis(tile, 0, -1)

				tile = cv.cvtColor(tile, cv.COLOR_RGBA2BGR)

				total_count += detect_pumpkins_hls(tile, idx)
				idx += 1


	print(f"Total count is {total_count}")

# Example usage
split_geotiff("figures/pumpkins_cropped.tif", "test/output_tiles", tile_size=512)
