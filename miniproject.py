import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import scipy


class miniproject:
    
    def __init__(self):
        self.annotated_image = None
        
    
    def open_image(self, path):
        image = cv.imread(path)
        return image
    
    
if __name__ == "__main__":
    project = miniproject()
    image = project.open_image('figures/pumpkin_annottated.JPG')
    cv.imshow("annotated", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    