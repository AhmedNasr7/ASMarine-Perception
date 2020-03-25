from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv2
from google.colab.patches import cv2_imshow

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    blurr = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(blurr):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)

    cv2.drawContours(img, squares, -1, (0, 255, 0), 3 )       
    return img

def find_circles(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5, 5), 0)

  circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,20,
                         param1=60,param2=40,minRadius=1,maxRadius=150)
  if circles is None:
    return img

  detected_circles = np.uint16(np.around(circles))
  for (x, y, r) in detected_circles[0, :]:
     cv2.circle(img, (x,y), r, (0,255,0), 6)
     img = cv2.putText(img, 'Circle', (x-15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
       
  return img 
 
def find_pentagon(img):
    blurr = cv2.GaussianBlur(img, (5, 5), 0)
    pentagon = []
    for gray in cv2.split(blurr):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.06*cnt_len, True)
                
                if len(cnt) == 5 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 5], cnt[(i+2) % 5] ) for i in xrange(5)])
                    
                    if max_cos < 0.5:
                        pentagon.append(cnt)
    cv2.drawContours(img, pentagon, -1, (0, 255, 0), 3 )          
    return img

def find_octagon(img):
    blurr = cv2.GaussianBlur(img, (5, 5), 0)
    octagon = []
    for gray in cv2.split(blurr):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.01*cnt_len, True)

                if len(cnt) == 8 and cv2.contourArea(cnt) >1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 8], cnt[(i+2) % 8] ) for i in xrange(8)])
                  
                    if max_cos < 0.9:
                        octagon.append(cnt)
    
    cv2.drawContours(img, octagon, -1, (0, 255, 0), 3 )     
    return img

def main():
    from glob import glob
    for fn in glob('/content/586.jpg'):
        img = cv2.imread(fn)
        img = find_circles(img)
        img = find_squares(img)
        img = find_octagon(img)
        img = find_pentagon(img)
        cv2_imshow(img)
        ch = cv2.waitKey()
        if ch == 27:
            break

    #print('Done')


if __name__ == '__main__':
    #print(__doc__)
    main()
    cv2.destroyAllWindows()