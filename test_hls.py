import cv2 as cv
import numpy as np

image = cv.imread('figures/pumpkins_cropped.tif')
hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
_, _, S = cv.split(hls)

cv.imwrite('test/S_orig.png', S)

_, mask = cv.threshold(S, 230, 255, cv.THRESH_BINARY_INV)

cv.imwrite('test/S.png', mask)

indices = np.where(mask > 0)
extracted_pixels = image[indices[0], indices[1]]

cv.imwrite("test/mask.png", mask)
cv.imwrite("test/mask_and.png", cv.bitwise_and(image, image, mask=mask))

# Calculate the mean and covariance matrix of the extracted pixels
mean = np.mean(extracted_pixels, axis=0)
mean = np.array([mean[2], mean[1], mean[0]])

covar, _ = cv.calcCovarMatrix(extracted_pixels, mean, cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE)

# Calculate the Mahalanobis distance
pixels = np.reshape(image, (-1, 3))
diff = pixels - np.repeat([mean], pixels.shape[0], axis=0)
inv_cov = np.linalg.inv(covar)
moddotproduct = diff * (diff @ inv_cov)
mahalanobis_dist = np.sum(moddotproduct, axis=1)
mahalanobis_mask = np.reshape(mahalanobis_dist, (image.shape[0], image.shape[1])).astype(np.uint8)

cv.imwrite('test/mahalanobis_dist.png', mahalanobis_mask)

# Apply a threshold to the mahalanobis_mask
_, mahalanobis_mask = cv.threshold(mahalanobis_mask, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# Save the mahalanobis_mask to inspect it
cv.imwrite('test/mahalanobis_mask.png', mahalanobis_mask)

contours, hierarchy = cv.findContours(mahalanobis_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Draw a circle above the center of each of the detected contours.
for contour in contours:
	M = cv.moments(contour)
	if M['m00'] != 0:
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])
		cv.circle(image, (cx, cy), 5, (0, 0, 255), 1)

print("Number of detected balls: %d" % len(contours))

cv.imwrite('test/circle.png', image)

