import cv2
import numpy as np


path= "Y:\\Pyth\\"
img = cv2.imread(path+"300.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imshow('edges', edges)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 200) #parametric coordinates (r, and theta)

i=0
arr = np.array([])

for r,theta in lines[i]:

      angle=[90-theta *(180/np.pi)]
      arr = np.append(arr, (angle))
      i += 1

print (arr)