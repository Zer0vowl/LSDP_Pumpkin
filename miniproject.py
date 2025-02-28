import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import scipy
from enum import Enum

class ColorSpace(Enum):
	CIELAB = 1
	HLS = 2
	HSV = 3
	RGB = 4

class miniproject:

	def __init__(self, tif_image = None, refference_image = None, refference_annotated=None):
		self.ref_image_annotated = None
		self.image = None
		self.mask = None
		self.ref_image = None
		self.colorspace = ColorSpace.RGB

		if tif_image:
			self.image = self.open_image(tif_image)

		if refference_image:
			self.ref_image = self.open_image(refference_image)
			self.ref_image = cv.GaussianBlur(self.ref_image, (5, 5), 0)

		if refference_annotated:
			self.ref_image_annotated = self.open_image(refference_annotated)


	def open_image(self, path):
		image = cv.imread(path)
		return image


	def change_color_space(self, color_space):
		self.colorspace = color_space

		if color_space == ColorSpace.CIELAB:
			self.ref_image = cv.cvtColor(self.ref_image, cv.COLOR_BGR2LAB)
		elif color_space == ColorSpace.HLS:
			self.ref_image = cv.cvtColor(self.ref_image, cv.COLOR_BGR2HLS)
		elif color_space == ColorSpace.HSV:
			self.ref_image = cv.cvtColor(self.ref_image, cv.COLOR_BGR2HSV)
		elif color_space == ColorSpace.RGB:
			self.ref_image = cv.cvtColor(self.ref_image, cv.COLOR_BGR2RGB)
		else:
			print("Imcorrect color space")


	def threshold(self, image, lower_bound, upper_bound):
		self.mask = cv.inRange(image, lower_bound, upper_bound)
		return self.mask


	def mahalanobis_distance_mask(self):
		# Threshold the annotated image to get the mask
		red_lower = (0, 0, 200)
		red_upper = (100, 100, 255)
		annotated_mask = self.threshold(self.ref_image_annotated, red_lower, red_upper)

		# Extract the pixels from the annotated mask
		indices = np.where(annotated_mask > 0)
		extracted_pixels = project.ref_image[indices[0], indices[1]]

		# Calculate the mean and covariance matrix of the extracted pixels
		mean = np.mean(extracted_pixels, axis=0)
		covar, _ = cv.calcCovarMatrix(extracted_pixels, mean, cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE)

		# Calculate the Mahalanobis distance
		pixels = np.reshape(self.ref_image, (-1, 3))
		diff = pixels - np.repeat([mean], pixels.shape[0], axis=0)
		inv_cov = np.linalg.inv(covar)
		moddotproduct = diff * (diff @ inv_cov)
		mahalanobis_dist = np.sum(moddotproduct, axis=1)
		mahalanobis_distance_image = np.reshape(mahalanobis_dist, (self.ref_image.shape[0], self.ref_image.shape[1]))

		# The image contains values between 0 and 255. Extract only the values below 10. Set them to 255. The rest to 0.
		cv.imwrite("figures/output/mahalanobis_distance_image.jpg", mahalanobis_distance_image)
		cv.threshold(mahalanobis_distance_image, 10, 255, cv.THRESH_BINARY_INV, mahalanobis_distance_image)

		cv.imwrite("figures/output/mahalanobis_distance_image_thrashold.jpg", mahalanobis_distance_image)
		return mahalanobis_distance_image.astype(np.uint8)


if __name__ == "__main__":
	project = miniproject('./figures/pumpkins_cropped.tif', 'figures/EB-02-660_0595_0068.JPG','figures/pumpkin_annottated.JPG')
	if project.image is None:
		Warning("Image not loaded")
		exit(1)

	project.change_color_space(ColorSpace.HLS)

	mahalanobis_mask = project.mahalanobis_distance_mask()
	masked_image = cv.bitwise_and(project.ref_image, project.ref_image, mask=mahalanobis_mask)

	# Only take BGR values
	mean_color = cv.mean(project.ref_image, mask=mahalanobis_mask)
	mean, stddev = cv.meanStdDev(project.ref_image, mask=mahalanobis_mask)

	# Convert from 2D array to tuple
	mean = tuple(mean.flatten())
	stddev = tuple(stddev.flatten())
	stddev_half = [a/2 for a in stddev]

	# Setup the lower and upper bounds for the mask.
	# The color is dependent on the mean and standard deviation of the values created by the annotated image.
	gain = 4
	low_orange = np.array([a - b for a, b in zip(mean, stddev)])
	upper_orange = np.array([a + b for a, b in zip(mean, stddev)])

	global_mask = project.threshold(project.image, low_orange, upper_orange)
	cv.imwrite("figures/output/global_mask.jpg", global_mask)

	global_mask = cv.dilate(global_mask, None, iterations=1)
	cv.imwrite("figures/output/global_mask_dilated.jpg", global_mask)

	out = cv.bitwise_and(project.image, project.image, mask=global_mask)
	cv.imwrite("figures/output/out.jpg", out)

	cv.imwrite("figures/output/mask.jpg", project.mask)
	cv.imwrite("figures/output/masked_image.jpg", masked_image)

	""" cv.imshow("annotated", masked_image) """
	""" cv.waitKey(0) """
	cv.destroyAllWindows()

	contours, hierarchy = cv.findContours(global_mask, cv.RETR_TREE,
			cv.CHAIN_APPROX_SIMPLE)

	# Draw a circle above the center of each of the detected contours.
	for contour in contours:
		M = cv.moments(contour)
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])
		cv.circle(project.image, (cx, cy), 5, (0, 0, 255), 1)

	cv.imwrite("figures/output/detected_pumpkins.jpg", project.image)

	print("Number of detected balls: %d" % len(contours))
