import sys
sys.path.remove(sys.path[1])

import cv2 
import numpy as np
import math
import os
import time

imgs = [525,530,541,543,545,546,547,548,549,550]


def get_angle_and_position(im):
    t0 = time.time()
    img_name = 'images/'+str(im)+'_enhanced.jpg'
    img_name = '/'.join(os.path.abspath(__file__).split('/')[:-1])+'/'+img_name



    # Read image given by user
    image = cv2.imread(img_name)
#        image = cv2.resize(image,(600,600))
    if image is None:
        exit(0)
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1 # Simple contrast control
    beta = 70   # Simple brightness control
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)

    hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)         #5ms max
    
    orange_low = np.array([4, 65, 50])
    orange_high = np.array([30, 255, 255])

    o_mask = cv2.inRange(hsv, orange_low, orange_high)      #2ms max
    new_image = cv2.bitwise_and(new_image,new_image,mask=o_mask)
    
    
    tx = time.time()
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)                   #7ms
    tx = time.time()
    ret, thresh = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)      #7ms

    regions = cv2.connectedComponentsWithStats(thresh)      #4ms
    ind, i= 1,1
    area = 0
    new_thresh = 0
#################################### less than 1 ms ###############
    if regions[0]>1:
        for region in regions[2][1:]:
            if region[-1] > area :
                ind = i
                area = region[-1]
            i+=1
        new_thresh , centroid = get_roi(thresh,regions,ind,im)
##############################################################################################
    else:
        return None,None
    contours, _hierarchy = cv2.findContours(new_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)         ## less than 1ms


    angles =[]   
    lines_list = []
    for i in range(len(contours)):
        if(len(contours[i])) >=5:
#            (x,y),( MA,ma),angle = cv2.fitEllipse(contours[i])
#            angles.append(angle)
            lines_list.append(find_and_draw_lines(new_thresh,contours))
#        temp = np.zeros(image.shape,image.dtype)
    for lines_array in lines_list:
        max_len , max_ang = 0,0
        for line in lines_array:
#            cv2.line(temp,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(255,0,0),thickness=2)
            ang , length = angle_and_length(line)
            if length > max_len:
                max_len = length
                max_ang = ang





#    print('Angles of each contour (in degrees): ' + str(angles))
    print('angle = %.2f'%max_ang)
#    cv2.imshow('New Image__'+str(im), new_image)
#    cv2.imshow('lines__'+str(im),temp)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    return max_ang,centroid


def angle_and_length(line):
    line = line[0]
    slope = (line[3]-line[1])/(line[2]-line[0])
    length = cv2.norm(line[0:2],line[2:])
    return (180-round(math.degrees(math.atan(slope)),2))%181,round(length,2)
    

def get_roi(image,regions,index,im_num):
    im_num = im_num
    ty,by= regions[2][index,0] , regions[2][index,0]+regions[2][index,2]
    lx,rx= regions[2][index,1] , regions[2][index,1]+regions[2][index,3]+1 
    new = np.zeros(image.shape,image.dtype)
    new[lx:rx,ty:by] = image[lx:rx,ty:by]
    return new , regions[3][index]


def find_and_draw_lines(image,contours):
    temp = cv2.drawContours(np.zeros((image.shape[0],image.shape[1]),image.dtype)\
                ,contours,-1,(255,255,255),1)
    lines = cv2.HoughLinesP(temp, 1 ,np.pi/180 , threshold=18,minLineLength = 5,maxLineGap=20)
    return lines


if __name__ == "__main__":
    for im in imgs:        
        get_angle_and_position(im)