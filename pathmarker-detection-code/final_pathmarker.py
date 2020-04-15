from __future__ import print_function
from builtins import input
import cv2 
import numpy as np
import argparse
import math

def get_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY


def get_angle(p1, p2, p3 ):
    return math.atan2(p1[1] - p2[1] - p3[1] , p1[0] - p2[0] - p3[0]) * 180/math.pi





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

orange_low = np.array([5, 50, 50])
orange_high = np.array([40, 255, 255])

o_mask = cv2.inRange(hsv, orange_low, orange_high)

cp_image[o_mask == 0] = [0, 0, 0]

gray = cv2.cvtColor(cp_image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)

contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

for cnt in contours:
    (x,y),(MA,ma),angle = cv2.fitEllipse(cnt) 
    print(angle)   

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