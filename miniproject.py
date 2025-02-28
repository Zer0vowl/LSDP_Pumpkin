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
		print(self.mean)


if __name__ == "__main__":
	project = miniproject('figures/EB-02-660_0595_0068.JPG','figures/pumpkin_annottated.JPG')

	project.change_color_space(COLOR_SPACE.HLS)

	red_lower = (0, 0, 200)
	red_upper = (100, 100, 255)
	project.create_mask(project.annotated_image, red_lower, red_upper)


	masked_image = None
	masked_image = cv.bitwise_and(project.image, project.image, mask=project.mask)
	
	print(type(masked_image))
	mean_color = cv.mean(project.image, mask=project.mask)  # Only take BGR values
 
	mean, stddev = cv.meanStdDev(project.image, mask=project.mask)

	# Convert from 2D array to tuple
	mean = tuple(mean.flatten())
	stddev = tuple(stddev.flatten())
	stddev_half = [a/2 for a in stddev]

	#low_orange = tuple(a - b for a, b in zip(mean, std)

	print(stddev)
	print(stddev_half)

	cv.imwrite("figures/output/mask.jpg", project.mask)
	cv.imwrite("figures/output/masked_image.jpg", masked_image)

	cv.imshow("annotated", masked_image)
	cv.waitKey(0)
	cv.destroyAllWindows()
