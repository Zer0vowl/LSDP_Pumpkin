import cv2 as cv
import numpy as np

def detect_pumpkins_hls(image, idx):
	hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
	_, _, S = cv.split(hls)

	cv.imwrite(f"test/S_orig_{idx}.png", S)

	_, mask = cv.threshold(S, 190, 255, cv.THRESH_BINARY)

	cv.imwrite(f"test/S_{idx}.png", S)

	indices = np.where(mask > 0)
	if len(indices[0]) < 2:
		return 0

	extracted_pixels = image[indices[0], indices[1]]

	cv.imwrite(f"test/mask_{idx}.png", mask)
	tmp = cv.bitwise_and(image, image, mask=mask)
	cv.imwrite(f"test/mask_and_{idx}.png", tmp)

# Calculate the mean and covariance matrix of the extracted pixels
	mean = np.mean(extracted_pixels, axis=0)

	covar, _ = cv.calcCovarMatrix(extracted_pixels, mean, cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE)

# Calculate the Mahalanobis distance
	pixels = np.reshape(image, (-1, 3))
	diff = pixels - np.repeat([mean], pixels.shape[0], axis=0)
	inv_cov = np.linalg.inv(covar + np.eye(covar.shape[0]) * 1e-6)  # Regularization for stability
	moddotproduct = diff * (diff @ inv_cov)
	mahalanobis_dist = np.sum(moddotproduct, axis=1)
	mahalanobis_mask = np.reshape(mahalanobis_dist, (image.shape[0], image.shape[1]))
	mahalanobis_mask = 255 * mahalanobis_mask / np.max(mahalanobis_mask)
	mahalanobis_mask = mahalanobis_mask.astype(np.uint8)

	cv.imwrite(f"test/mahalanobis_dist_{idx}.png", mahalanobis_mask)

# Apply a threshold to the mahalanobis_mask
	_, mahalanobis_mask = cv.threshold(mahalanobis_mask, 0, 255, cv.THRESH_BINARY_INV)

# Save the mahalanobis_mask to inspect it
	cv.imwrite(f"test/mahalanobis_mask_{idx}.png", mahalanobis_mask)

	contours, hierarchy = cv.findContours(mahalanobis_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Draw a circle above the center of each of the detected contours.
	for contour in contours:
		M = cv.moments(contour)
		if M['m00'] != 0:
			cx = int(M['m10'] / M['m00'])
			cy = int(M['m01'] / M['m00'])
			cv.circle(image, (cx, cy), 5, (0, 0, 255), 1)

	print("Number of detected balls: %d" % len(contours))

	cv.imwrite(f"test/circle_{idx}.png", image)
	return len(contours)

