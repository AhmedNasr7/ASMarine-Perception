import cv2
import math
import numpy as np



def pyramid_reconstruct(pyramid):

    level = len(pyramid)
    for i in range(level - 1, 0, -1):

        tmpPyr = cv2.resize(pyramid[i], (pyramid[i - 1].shape[1], pyramid[i - 1].shape[0]), 0, 0, cv2.INTER_LINEAR)
        #Core.add(pyramid[i - 1], tmpPyr, pyramid[i - 1]);
        pyramid[i - 1] = cv2.add(tmpPyr, pyramid[i - 1])
    
    return pyramid[0];




def gaussian_pyramid(img, level):

  h = 1/16* np.array([1, 4, 6, 4, 1])
  filt = np.dot(h, np.transpose(h))
  filtered = cv2.filter2D(img, -1, filt, borderType=cv2.BORDER_REPLICATE)

  gaussPyr = [0] * level

  gaussPyr[0] = np.copy(filtered)
  tmpImg = np.copy(img)

  for i in range(1, level):
    tmpImg = cv2.resize(tmpImg, (img.shape[0], img.shape[1]), 0.5, 0.5, cv2.INTER_LINEAR)
    tmp = cv2.filter2D(tmpImg, -1, filt)
    gaussPyr[i] = np.copy(tmp)

  return gaussPyr
        

def laplacian_pyramid(img, gpA, level):

  h = 1/16* np.array([1, 4, 6, 4, 1])
  filt = np.dot(h, np.transpose(h))

  filtered = cv2.filter2D(img.astype('uint8'), -1, filt, borderType=cv2.BORDER_REPLICATE)

  lapPyr = [0] * level

  lapPyr[0] = np.copy(img)
  tmpImg = np.copy(img)

  for i in range(1, level):
    tmpImg = cv2.resize(tmpImg.astype('uint8'), (img.shape[0], img.shape[1]), 0.5, 0.5, cv2.INTER_LINEAR)
    lapPyr[i] = np.copy(tmpImg)
  
  # calculate the DoG
  for i in range(1, level - 1):
    tmpPyr = cv2.resize(lapPyr[i + 1], (lapPyr[i].shape[1], lapPyr[i].shape[0]) , 0, 0, cv2.INTER_LINEAR)

    lapPyr[i] = cv2.subtract(lapPyr[i], tmpPyr)

  return lapPyr



