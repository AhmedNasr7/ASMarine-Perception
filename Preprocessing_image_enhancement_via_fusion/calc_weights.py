import cv2
import math
import numpy as np 
from saliency_detection import *


pi = 3.14


def calc_weights(img1, img2): # calculate weights of 2 images


	#input1
	lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)

	R1 = np.uint32(lab1[:, :, 0]) / 255.

	# calculate laplacian contrast weight

	WL1 = abs (cv2.Laplacian(R1,cv2.CV_64F, borderType=cv2.BORDER_REPLICATE))

	#calculate Local contrast weight

	h = 1/16 * np.array([1, 4, 6, 4, 1])

	kernel = np.dot(h, np.transpose(h)) # building kernel, equivalent to h'*h in matlab

	WC1 = cv2.filter2D(R1, -1, kernel, borderType=cv2.BORDER_REPLICATE)

	q=np.where(WC1 > (pi/2.75))

	WC1[q] = pi/2.75

	WC1 = (R1 - WC1) ** 2

	WS1 = saliency_detection(img1)

	# calculate the exposedness weight

	sigma = 0.25

	aver = 0.5

	WE1 = np.exp(-(R1 - aver) ** 2 / (2*sigma ** 2))

	# input2

	lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)

	R2 = np.uint32(lab2[:, :, 0]) / 255.

	#calculate laplacian contrast weight

	WL2 =abs (cv2.Laplacian(R1,cv2.CV_64F,borderType=cv2.BORDER_REPLICATE)) 

	#calculate Local contrast weight
	WC2 = cv2.filter2D(R1, -1, kernel, borderType=cv2.BORDER_REPLICATE)

	t = np.where(WC2 > (pi/2.75))

	WC2[t] = pi/2.75

	WC2 = (R2 - WC2) ** 2
	# calculate the saliency weight

	WS2 = saliency_detection(img2)

	# calculate the exposedness weighs

	sigma = 0.25
	aver = 0.5

	WE2 = np.exp(-(R2 - aver) ** 2 / (2*sigma ** 2))

	# calculate the normalized weight

	W1 = (WL1 + WC1 + WS1 + WE1) / (WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2)

	W2 = (WL2 + WC2 + WS2 + WE2)/(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2)


	return W1, W2