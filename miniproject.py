import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import scipy
from enum import Enum

class COLOR_SPACE(Enum):
	CIELAB = 1
	HLS = 2
	HSV = 3
	RGB = 4

class miniproject:

	def __init__(self, path_image = None, path_annotated=None):
		self.annotated_image = None
		self.image = None
		self.mask = None

		self.mean = 0
		self.std = 0

		if path_image:
			self.image = self.open_image(path_image)
			self.image = cv.GaussianBlur(self.image, (5, 5), 0)
		if path_annotated:
			self.annotated_image = self.open_image(path_annotated)

	def open_image(self, path):
		image = cv.imread(path)
		return image

	def change_color_space(self, color_space):
		if color_space == COLOR_SPACE.CIELAB:
			self.image = cv.cvtColor(self.image, cv.COLOR_BGR2LAB)
		elif color_space == COLOR_SPACE.HLS:
			self.image = cv.cvtColor(self.image, cv.COLOR_BGR2HLS)
		elif color_space == COLOR_SPACE.HSV:
			self.image = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
		elif color_space == COLOR_SPACE.RGB:
			self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
		else:
			print("Imcorrect color space")


	def create_mask(self, image, lower_bound, upper_bound):
		self.mask = cv.inRange(image, lower_bound, upper_bound)

		self.mean, self.std = cv.meanStdDev(image, self.mask)
		return self.mask
		""" print(self.mean) """


if __name__ == "__main__":
	project = miniproject('figures/EB-02-660_0595_0068.JPG','figures/pumpkin_annottated.JPG')
	if project.image is None:
		Warning("Image not loaded")
		exit(1)

	""" project.change_color_space(COLOR_SPACE.HLS) """

	red_lower = (0, 0, 200)
	red_upper = (100, 100, 255)
	annotated_mask = project.create_mask(project.annotated_image, red_lower, red_upper)


	masked_image = None
	masked_image = cv.bitwise_and(project.image, project.image, mask=annotated_mask)

	""" print(type(masked_image)) """
	mean_color = cv.mean(project.image, mask=project.mask)  # Only take BGR values

	mean, stddev = cv.meanStdDev(project.image, mask=project.mask)

	# Convert from 2D array to tuple
	mean = tuple(mean.flatten())
	stddev = tuple(stddev.flatten())
	stddev_half = [a/2 for a in stddev]

	# Setup the lower and upper bounds for the mask.
	# The color is dependent on the mean and standard deviation of the values created by the annotated image.
	gain = 4
	low_orange = np.array([a - b for a, b in zip(mean, stddev*gain)])
	upper_orange = np.array([a + b for a, b in zip(mean, stddev*gain)])
	print(low_orange, upper_orange)

	global_mask = project.create_mask(project.image, low_orange, upper_orange)
	cv.imwrite("figures/output/global_mask.jpg", global_mask)

	closed_image = cv.dilate(global_mask, None, iterations=4)
	cv.imwrite("figures/output/global_mask_dilated.jpg", closed_image)

	""" kernel = np.ones((10, 10), np.uint8) """
	""" closed_image = cv.morphologyEx(global_mask, cv.MORPH_CLOSE, kernel) """
	""" cv.imwrite("figures/output/closed_image.jpg", closed_image) """

	out = cv.bitwise_and(project.image, project.image, mask=closed_image)
	cv.imwrite("figures/output/out.jpg", out)

	cv.imwrite("figures/output/mask.jpg", project.mask)
	cv.imwrite("figures/output/masked_image.jpg", masked_image)

	""" cv.imshow("annotated", masked_image) """
	""" cv.waitKey(0) """
	cv.destroyAllWindows()

	contours, hierarchy = cv.findContours(closed_image, cv.RETR_TREE,
			cv.CHAIN_APPROX_SIMPLE)

	# Draw a circle above the center of each of the detected contours.
	for contour in contours:
		M = cv.moments(contour)
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])
		cv.circle(project.image, (cx, cy), 40, (0, 0, 255), 2)

	print("Number of detected balls: %d" % len(contours))
