import cv2
import numpy as np
import math

path= "Y:\\Pyth\\"
img = cv2.imread(path+"S1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imshow('edges', edges)

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10) #cartesian coordinates (x,y)

for line in lines:
    for x1, y1, x2, y2 in line:
        m = (y2 - y1) / (x2 - x1)
        angle = math.degrees(math.atan(m))

print(angle)
