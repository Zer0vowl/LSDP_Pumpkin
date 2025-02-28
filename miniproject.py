import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import scipy

class COLOR_SPACE(Enum):
	CIELAB = 1
	HSL = 2
	HSV = 3
	RGB = 4

class miniproject:
	
	def __init__(self, path_image = None, path_annotated=None):
		self.annotated_image = None
		self.image = None
		
		if path_image:
			self.image = self.open_image(path_image)
		if path_annotated:
			self.annotated_image = self.open_image(path_annotated)

	def open_image(self, path):
		image = cv.imread(path)
		return image

	def change_color_space(self, image, color_space):
		
		if color_space == COLOR_SPACE.CIELAB:
			image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
		elif color_space == COLOR_SPACE.HSL:
			image = cv.cvtColor(image, cv.COLOR_BGR2HLS)
		elif color_space == COLOR_SPACE.HSV:
			image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
		elif color_space == COLOR_SPACE.RGB:
			image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		else:
			print("Imcorrect color space")
			return None

		return image

	def create_mask(self, lower_band, upper_band):
		mask = cv.inRange(self.annotated_image, lower_band, upper_band)

		mean, std = cv.meanStdDev(self.image, mask = mask)
		print(mean)
	


if __name__ == "__main__":
	project = miniproject()
	image = project.open_image('figures/pumpkin_annottated.JPG')
	cv.imshow("annotated", image)
	cv.waitKey(0)
	cv.destroyAllWindows()
