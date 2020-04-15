from __future__ import print_function
from builtins import input
import cv2 
import numpy as np
import argparse
import math


img_name = 'images/pmarker3-f.jpg'


# Read image given by user
parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! tutorial.')
parser.add_argument('--input', help='Path to input image.', default=img_name)
args = parser.parse_args()
image = cv2.imread(cv2.samples.findFile(args.input))
if image is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
new_image = np.zeros(image.shape, image.dtype)
alpha = 1.0 # Simple contrast control
beta = 70   # Simple brightness control

# Do the operation new_image(i,j) = alpha*image(i,j) + beta
# Instead of these 'for' loops we could have used simply:
new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
# but we wanted to show you how to access the pixels :)
cp_image = np.copy(new_image)

hsv = cv2.cvtColor(cp_image, cv2.COLOR_BGR2HSV)

orange_low = np.array([6, 65, 50])
orange_high = np.array([30, 255, 255])

o_mask = cv2.inRange(hsv, orange_low, orange_high)

cp_image[o_mask == 0] = [0, 0, 0]

#img2 = cv2.bitwise_not(cp_image)

gray = cv2.cvtColor(cp_image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)

contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

angles =[]   
for i in range(len(contours)):
    if(len(contours[i])) >=5:
        (x,y),( MA,ma),angle = cv2.fitEllipse(contours[i])
        angles.append(angle) 
        #print(str(angle))

print('Angles of each contour (in degrees): ' + str(angles))


# Otsu's thresholding after Gaussian filtering
#blur = cv2.GaussianBlur(img,(3,3),0)
#ret3,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


kernel = np.ones((7,7),np.uint8)
#dilated = cv2.dilate(thresh,kernel,iterations = 20)
#eroded = cv2.erode(thresh,kernel,iterations = 1)


cv2.imshow('Original Image', image)
cv2.imshow('New Image', new_image)
cv2.imshow('mask',thresh)
# Wait until user press some key
cv2.waitKey(0)
