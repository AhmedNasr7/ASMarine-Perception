from __future__ import print_function
import sys
sys.path.remove(sys.path[1])

from builtins import input
import cv2 
import numpy as np
import argparse
import math
import os
import time

imgs = [525,530,541,543,545,546,547,548,549,550]


def main():
    for im in imgs:
#        img_name = 'images/pmarker1-f.jpg'
        
        t0 = time.time()
        img_name = 'images/'+str(545)+'_enhanced.jpg'
        img_name = '/'.join(os.path.abspath(__file__).split('/')[:-1])+'/'+img_name



        # Read image given by user
        parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! tutorial.')
        parser.add_argument('--input', help='Path to input image.', default=img_name)
        args = parser.parse_args()
        image = cv2.imread(cv2.samples.findFile(args.input))
        if image is None:
            print('Could not open or find the image: ', args.input)
            exit(0)
        new_image = np.zeros(image.shape, image.dtype)
        alpha = 2 # Simple contrast control
        beta = 0   # Simple brightness control
        # Do the operation new_image(i,j) = alpha*image(i,j) + beta
        # Instead of these 'for' loops we could have used simply:
        new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        # but we wanted to show you how to access the pixels :)
        cp_image = np.copy(new_image)

        hsv = cv2.cvtColor(cp_image, cv2.COLOR_BGR2HSV)

        orange_low = np.array([4, 65, 50])
        orange_high = np.array([30, 255, 255])

        o_mask = cv2.inRange(hsv, orange_low, orange_high)

        cp_image[o_mask == 0] = [0, 0, 0]

        #img2 = cv2.bitwise_not(cp_image)
        gray = cv2.cvtColor(cp_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
        t1 = time.time()
        regions = cv2.connectedComponentsWithStats(thresh)
        ind, i= 1,1
        area = 0
        new_thresh = thresh
#        print(type(regions[2][1][0]))
#        exit(0)
        if regions[0]>1:
            for region in regions[2][1:]:
                if region[-1] > area :
                    ind = i
                    area = region[-1]
                i+=1
            new_thresh = get_roi(thresh,regions,ind,im)
#            print('region selection time = ',time.time()-t1)
        contours, _hierarchy = cv2.findContours(new_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        t2 = time.time()
        angles =[]   
        lines_list = []
        for i in range(len(contours)):
            if(len(contours[i])) >=5:
                (x,y),( MA,ma),angle = cv2.fitEllipse(contours[i])
                angles.append(angle)
                lines_list.append(find_and_draw_lines(new_thresh,contours))
#                exit(0) 
                #print(str(angle))
        temp = np.zeros(image.shape,image.dtype)
#        print('finding lines and appending time : ',time.time()-t2)
        for lines_array in lines_list:
            for line in lines_array:
                cv2.line(temp,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(255,0,0),thickness=2)
                #print(slope_and_length(line))

        print('Angles of each contour (in degrees): ' + str(angles))


        # Otsu's thresholding after Gaussian filtering
        #blur = cv2.GaussianBlur(img,(3,3),0)
        #ret3,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        kernel = np.ones((7,7),np.uint8)
        #dilated = cv2.dilate(thresh,kernel,iterations = 20)
        #eroded = cv2.erode(thresh,kernel,iterations = 1)

#        print('total time = ',time.time()-t0)

        #cv2.imshow('Original Image__'+str(im), image)
        cv2.imshow('New Image__'+str(im), new_image)
        cv2.imshow('lines__'+str(im),temp)
        #cv2.imshow('mask__'+str(im),thresh)
        #cv2.imshow('new_mask__'+str(im),new_thresh)
        # Wait until user press some key
        cv2.waitKey()
        cv2.destroyAllWindows()


def slope_and_length(line):
    t0 = time.time()
    line = line[0]
    slope = (line[3]-line[1])/(line[2]-line[0])
    length = cv2.norm(line[0:2],line[2:])
 #   print('slope_and_length_time = ',time.time()-t0)
    return round(slope,2),round(length,2)
    

def get_roi(image,regions,index,im_num):
    t0 = time.time()
    im_num = im_num
    ty,by= regions[2][index,0] , regions[2][index,0]+regions[2][index,2]
    lx,rx= regions[2][index,1] , regions[2][index,1]+regions[2][index,3]+1 
    new = np.zeros(image.shape,image.dtype)
    new[lx:rx,ty:by] = image[lx:rx,ty:by]
#    print('get_roi time = ',time.time()-t0)
    return new


def find_and_draw_lines(image,contours):
    temp = cv2.drawContours(np.zeros((image.shape[0],image.shape[1]),image.dtype)\
                ,contours,-1,(255,255,255),1)
    lines = cv2.HoughLinesP(temp, 1 ,np.pi/180 , threshold=18,minLineLength = 5,maxLineGap=20)
    return lines


if __name__ == "__main__":
    main()