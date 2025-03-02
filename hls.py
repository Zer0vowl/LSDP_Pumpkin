import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import os

OUT_DIR = "./figures/kenya/output/"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}")

class Mahalanobis():
	def __init__(self, image, annotated_image):
		self.mean = None
		self.inv_cov = None
		self.ann_image_mask = None

		self.__createAnnotatedMask(annotated_image)
		pixels = self.__extractAnnotatedPixels(image)
		print(f"Selected {len(pixels)} pixels for analysis.")

		# Calculate the mean and covariance matrix of the extracted pixels
		self.mean = np.mean(pixels, axis=0)

		covar, _ = cv.calcCovarMatrix(pixels, self.mean, cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE)
		covar *= 100

		self.inv_cov = np.linalg.inv(covar)
		print(f"Mean: {self.mean}")
		print(f"Inv Cov: {self.inv_cov}")


	def __createAnnotatedMask(self, image):
		red_lower = (0, 0, 220)  # Adjusted for BGR (Blue, Green, Red)
		red_upper = (2, 2, 255)

		# Create annotation mask
		self.ann_image_mask = cv.inRange(image, red_lower, red_upper)
		cv.imwrite(OUT_DIR + "annotated_mask.jpg", self.ann_image_mask)


	def __extractAnnotatedPixels(self, image):
		selected_pixels = image[self.ann_image_mask > 0]
		return np.reshape(selected_pixels, (-1, 3))


	def calculateMahalanobisMask(self, image, index):
		# Calculate the Mahalanobis distance
		pixels = np.reshape(image, (-1, 3))
		means = np.repeat([self.mean], pixels.shape[0], axis=0)
		diff = pixels - means

		moddotproduct = diff * (diff @ self.inv_cov)
		mahalanobis_dist = np.sum(moddotproduct, axis=1)
		mahalanobis_mask = np.reshape(mahalanobis_dist, (image.shape[0], image.shape[1]))
		mahalanobis_mask = 255 * mahalanobis_mask / np.max(mahalanobis_mask)
		mahalanobis_mask = mahalanobis_mask.astype(np.uint8)
		os.path.exists(OUT_DIR + f"{index}") or os.makedirs(OUT_DIR + f"{index}")
		cv.imwrite(OUT_DIR + f"{index}/mahalanobis_dist.png", mahalanobis_mask)

		# Apply a threshold to the mahalanobis_mask
		_, mahalanobis_mask = cv.threshold(mahalanobis_mask, 1, 255, cv.THRESH_BINARY_INV)
		""" mahalanobis_mask = cv.erode(mahalanobis_mask, (800,700), iterations=60) """
		mahalanobis_mask = cv.dilate(mahalanobis_mask, (3, 3), iterations=1)

		# Save the mahalanobis_mask to inspect it
		cv.imwrite(OUT_DIR + f"{index}/mahalanobis_mask.png", mahalanobis_mask)
		return mahalanobis_mask



class ImageDelegator():
	def __init__(self, image_name, mahalanobis, tile_size):
		self.image = image_name
		self.tile_size = tile_size
		self.mahalanobis = mahalanobis
		self.sum = 0


	def execute(self):
		with rasterio.open(self.image) as src:
			width = src.width
			height = src.height
			num_tiles_x = width // self.tile_size
			num_tiles_y = height // self.tile_size

			idx = 0
			# Iterate over the image
			for i in range(num_tiles_x):
				for j in range(num_tiles_y):
					# Define the window
					window = Window(i*self.tile_size, j*self.tile_size, self.tile_size, self.tile_size)

					tile = src.read(window=window)
					tile = np.moveaxis(tile, 0, -1)
					tile = cv.cvtColor(tile, cv.COLOR_RGBA2BGR)

					os.path.exists(OUT_DIR + f"{idx}") or os.makedirs(OUT_DIR + f"{idx}")
					cv.imwrite(OUT_DIR + f"{idx}/test_tile.png", tile)

					self.__process(tile, idx)
					idx = idx + 1

		return self.sum


	def __process(self, tile, index):
		mah_mask = self.mahalanobis.calculateMahalanobisMask(tile, index)

		# Apply mask
		masked_image = cv.bitwise_and(tile, tile, mask=mah_mask)
		masked_image = masked_image.astype(np.uint8)
		cv.imwrite(OUT_DIR + f"{index}/masked.png", masked_image)

		# Find contours and draw circles
		contours, _ = cv.findContours(mah_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		for contour in contours:
			M = cv.moments(contour)
			if M['m00'] != 0:
				cx = int(M['m10'] / M['m00'])
				cy = int(M['m01'] / M['m00'])
				cv.circle(tile, (cx, cy), 5, (0, 0, 255), 1)

		# Save the final processed tile
		cv.imwrite(OUT_DIR + f"detected_{index}.png", tile)

		print(f"Tile ({index}) - Detected objects: {len(contours)}")

		self.sum = self.sum + len(contours)


def oklab(image, idx):
	# Convert the image to Oklab
	oklab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
	l, a, b = cv.split(oklab_image)

	# Calculate the mean and standard deviation of the saturation channel
	cv.imwrite(OUT_DIR + f"{idx}/l.png", l)
	cv.imwrite(OUT_DIR + f"{idx}/a.png", a)
	cv.imwrite(OUT_DIR + f"{idx}/b.png", b)
	return l, a, b


# Load images
KENYA_DIR = "./figures/kenya/annotation/"
image = cv.imread(KENYA_DIR + "tile_6_18.png")
ann_image = cv.imread(KENYA_DIR + "tile_6_18_annotated.png")

assert image is not None, "Image is None"
assert ann_image is not None, "Annotated image is None"

mahalanobis = Mahalanobis(image, ann_image)

index = 1
_, _, b = oklab(image, index)
extened_b = extend_histogram(b, 130, 155)
cv.imwrite(OUT_DIR + f"{index}/extended_b.png", extened_b)

_, b_mask = cv.threshold(extened_b, 5, 255, cv.THRESH_BINARY_INV)

# Morphological filtering the image
kernel_size = 10
circular_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
closed_image = cv.morphologyEx(b_mask, cv.MORPH_CLOSE, circular_kernel)

cv.imwrite(OUT_DIR + f"{index}/b_mask.png", closed_image.astype(np.uint8))

# Locate contours.
contours, _ = cv.findContours(closed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw a circle above the center of each of the detected contours.
for contour in contours:
	if cv.contourArea(contour) > 0:
		M = cv.moments(contour)
		if M['m00'] != 0:
			cx = int(M['m10'] / M['m00'])
			cy = int(M['m01'] / M['m00'])
			cv.circle(image, (cx, cy), 40, (0, 0, 255), 5)

cv.imwrite(OUT_DIR + f"{index}/detected.png", image)






""" mah_mask = mahalanobis.calculateMahalanobisMask(image, index) """
""" masked_image = cv.bitwise_and(image, image, mask=mahalanobis.ann_image_mask) """
""" masked_image = masked_image.astype(np.uint8) """
""" cv.imwrite(OUT_DIR + f"{index}/masked.png", masked_image) """
""""""
""" contours, _ = cv.findContours(mah_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) """
""" for contour in contours: """
""" 	M = cv.moments(contour) """
""" 	if M['m00'] != 0: """
""" 		cx = int(M['m10'] / M['m00']) """
""" 		cy = int(M['m01'] / M['m00']) """
""" 		cv.circle(image, (cx, cy), 5, (0, 0, 255), 1) """
""""""
""" # Save the final processed tile """
""" cv.imwrite(OUT_DIR + f"{index}/detected.png", image) """
""""""
""" print(f"Tile ({index}) - Detected objects: {len(contours)}") """
""" delegator = ImageDelegator("./figures/pumpkins_cropped.tif", mahalanobis, 512) """
""" sum = delegator.execute() """
""" print(f"Overall pumpkins detected: {sum}") """


# Mean: [ 90.75866496 179.53594352 246.34338896]

