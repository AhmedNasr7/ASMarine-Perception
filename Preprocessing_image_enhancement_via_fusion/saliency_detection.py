import cv2
import math
import numpy as np


def saliency_detection(img):

	filtered = cv2.GaussianBlur(img,(3, 3), 0)

	lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2Lab)	

	l, a, b = cv2.split(lab)

	lm = np.mean(np.mean(l))
	am = np.mean(np.mean(a))
	bm = np.mean(np.mean(b))


	saliency_map = (l-lm) ** 2 + (a-am) ** 2 + (b-bm) ** 2

	cv2.normalize(saliency_map, saliency_map,0, 255, cv2.NORM_MINMAX)

	saliency_map = saliency_map.astype('uint8')


	return saliency_map



