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

img = cv2.imread(img_name)
image = np.copy(img)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

orange_low = np.array([5, 50, 50])
orange_high = np.array([40, 255, 255])

o_mask = cv2.inRange(hsv, orange_low, orange_high)

image[o_mask == 0] = [0, 0, 0]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

kernel = np.ones((7,7),np.uint8)
eroded = cv2.erode(thresh,kernel,iterations = 1)




cv2.imshow('original', img)
cv2.imshow('s', thresh)



print(time.time() - t0)

#cv2.imshow('s', img)
#cv2.imwrite('pmarker-3s.jpg', segmented_image)
cv2.waitKey(0)



