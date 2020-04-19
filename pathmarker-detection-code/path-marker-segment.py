import sys
sys.path.remove(sys.path[1])

import cv2
import math
import numpy as np
import time
import os




img_name = 'images/pmarker2-f.jpg'
img_name = '/'.join(os.path.abspath(__file__).split('/')[:-1])+'/'+img_name


t0 = time.time()

image = cv2.imread(img_name)


# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_vals = image.reshape((-1,3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)

k = 2
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

print(time.time() - t0)

cv2.imshow('s', segmented_image)
cv2.imwrite('pmarker-3s.jpg', segmented_image)
cv2.waitKey(0)



