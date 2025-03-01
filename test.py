import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def create_mask(image, lower_bound, upper_bound):
    mask = cv.inRange(image, lower_bound, upper_bound)
    return mask

def calculate_mahalanobis_bounds(pixels, threshold=3.0):
    mean = np.mean(pixels, axis=0)
    cov = np.cov(pixels.T)

    # Compute inverse covariance matrix
    inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)  # Regularization for stability

    # Compute Mahalanobis distance for each pixel
    distances = np.array([np.sqrt((p - mean) @ inv_cov @ (p - mean).T) for p in pixels])

    # Define bounds based on threshold
    mask = distances < threshold
    filtered_pixels = pixels[mask]

    lower_bound = np.min(filtered_pixels, axis=0)
    upper_bound = np.max(filtered_pixels, axis=0)

    return lower_bound.astype(np.uint8), upper_bound.astype(np.uint8)

# Configuration
use_hls = True  # Change to False to use BGR

# Load images
image = cv.imread("figures/EB-02-660_0595_0068.JPG")
ann_image = cv.imread("figures/pumpkin_annotated.JPG")

# Define initial annotation mask thresholds
if use_hls:
    image = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    ann_image = cv.cvtColor(ann_image, cv.COLOR_BGR2HLS)
    red_lower = (0, 120, 150)  # Adjusted for HLS (Hue, Lightness, Saturation)
    red_upper = (0, 255, 255)
else:
    red_lower = (0, 0, 220)  # Adjusted for BGR (Blue, Green, Red)
    red_upper = (2, 2, 255)

# Create annotation mask
ann_image_mask = create_mask(ann_image, red_lower, red_upper)
cv.imwrite("test_mask.jpg", ann_image_mask)

selected_pixels = image[ann_image_mask > 0]
print(f"Selected {len(selected_pixels)} pixels for analysis.")

# Compute Mahalanobis-based bounds
threshold = 3
lower_bound, upper_bound = calculate_mahalanobis_bounds(selected_pixels, threshold)
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

final_mask = create_mask(image, lower_bound, upper_bound)
final_mask = cv.erode(final_mask, None, iterations=1)
final_mask = cv.dilate(final_mask, None, iterations=1)
cv.imwrite("test_final_mask.jpg", final_mask)

masked_image = cv.bitwise_and(image, image, mask=final_mask)
if use_hls:
    masked_image = cv.cvtColor(masked_image, cv.COLOR_HLS2BGR)
cv.imwrite("test_masked_image.jpg", masked_image)

contours, hierarchy = cv.findContours(final_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Draw a circle above the center of each of the detected contours.
for contour in contours:
	M = cv.moments(contour)
	cx = int(M['m10'] / M['m00'])
	cy = int(M['m01'] / M['m00'])
	cv.circle(image, (cx, cy), 5, (0, 0, 255), 1)

if use_hls:
    image = cv.cvtColor(image, cv.COLOR_HLS2BGR)
cv.imwrite("detected_pumpkins.jpg", image)
print("Number of detected pumpkins: %d" % len(contours))

full_image = cv.imread("figures/pumpkins_cropped.tif")

if use_hls:
	full_image = cv.cvtColor(full_image, cv.COLOR_BGR2HLS)

final_mask = create_mask(full_image, lower_bound, upper_bound)
# final_mask = cv.erode(final_mask, None, iterations=1)
final_mask = cv.dilate(final_mask, None, iterations=1)
cv.imwrite("test_final_mask_full.jpg", final_mask)

masked_image = cv.bitwise_and(full_image, full_image, mask=final_mask)
if use_hls:
    masked_image = cv.cvtColor(masked_image, cv.COLOR_HLS2BGR)
cv.imwrite("test_masked_image_full.jpg", masked_image)

contours, hierarchy = cv.findContours(final_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Draw a circle above the center of each of the detected contours.
for contour in contours:
	M = cv.moments(contour)
	cx = int(M['m10'] / M['m00'])
	cy = int(M['m01'] / M['m00'])
	cv.circle(full_image, (cx, cy), 5, (0, 0, 255), 1)

if use_hls:
    full_image = cv.cvtColor(full_image, cv.COLOR_HLS2BGR)
cv.imwrite("detected_pumpkins_full.jpg", full_image)
print("Number of detected pumpkins: %d" % len(contours))