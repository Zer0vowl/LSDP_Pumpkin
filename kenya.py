import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import os


def extend_histogram(image, min_val, max_val):
	assert min_val < max_val, "Min value must be less than max value"
	assert len(image.shape) == 2, "Image must be a grayscale image"

	stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
	return stretched

TILE_SIZE = 2048

# Load images
KENYA_DIR = "./figures/kenya/"
TIFF_IMAGE = "figures/20250201_eBee_OlPejeta_survey.tif"

with rasterio.open(TIFF_IMAGE) as dataset:
	img_width, img_height = dataset.width, dataset.height

	total_count = 0

	num_tiles_x = img_width // TILE_SIZE
	num_tiles_y = img_height // TILE_SIZE

	index = 0
	# Iterate through the image in tile_size steps
	for i in range(num_tiles_x):
		for j in range(num_tiles_y):
			os.makedirs(KENYA_DIR + f"{index}", exist_ok=True)

			window = Window(i*TILE_SIZE, j*TILE_SIZE, TILE_SIZE, TILE_SIZE)
			tile = dataset.read(window=window)
			tile = np.moveaxis(tile, 0, -1)

			tile = cv.cvtColor(tile, cv.COLOR_RGBA2BGR)
			lab = cv.cvtColor(tile, cv.COLOR_BGR2LAB)
			_, _, b = cv.split(lab)

			extened_b = extend_histogram(b, 130, 155)
			cv.imwrite(KENYA_DIR + f"{index}/extended_b.png", extened_b)

			_, b_mask = cv.threshold(extened_b, 5, 255, cv.THRESH_BINARY_INV)

			# Morphological filtering the image
			kernel_size = 10
			circular_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
			closed_image = cv.morphologyEx(b_mask, cv.MORPH_CLOSE, circular_kernel)

			cv.imwrite(KENYA_DIR + f"{index}/b_mask.png", closed_image.astype(np.uint8))

			# Locate contours.
			contours, _ = cv.findContours(closed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

			# Draw a circle above the center of each of the detected contours.
			for contour in contours:
				if cv.contourArea(contour) > 0:
					M = cv.moments(contour)
					if M['m00'] != 0:
						cx = int(M['m10'] / M['m00'])
						cy = int(M['m01'] / M['m00'])
						cv.circle(tile, (cx, cy), 40, (0, 0, 255), 5)

			cv.imwrite(KENYA_DIR + f"{index}/detected.png", tile)
			os.makedirs(KENYA_DIR + "detected", exist_ok=True)
			cv.imwrite(KENYA_DIR + f"detected/detected_{index}.png", tile)

			index += 1
