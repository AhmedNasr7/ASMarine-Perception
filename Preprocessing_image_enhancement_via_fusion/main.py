import cv2
import math
import numpy as np
import time
from pyramids import *
from white_balance import *
from saliency_detection import *
from calc_weights import *





if __name__ == '__main__':




    img_name = '15.jpg'



    t0 = time.time()
    img = cv2.imread(img_name)

    img1 = white_balance(img, 1)

    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)	


    # Apply CLAHE on RGB

    lab2 = np.copy(lab1)

    l, a, b = cv2.split(lab2)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(7,7))
    cl = clahe.apply(l)
    lab2_clahe = cv2.merge((cl,a,b))

    img2 = cv2.cvtColor(lab2_clahe, cv2.COLOR_Lab2BGR)

    W1, W2 = calc_weights(img1, img2)


    # calculate the gaussian 

    level = 5
    Weight1 = gaussian_pyramid(W1, level)
    Weight2 = gaussian_pyramid(W2, level)

    #calculate the laplacian pyramid
    #input1
    R1 = laplacian_pyramid(img1[:, :, 0], Weight1, level)
    G1 = laplacian_pyramid(img1[:, :, 1], Weight1, level)
    B1 = laplacian_pyramid(img1[:, :, 2], Weight1, level)

    #input2
    R2 = laplacian_pyramid(np.uint32(np.uint32(img2[:, :, 0])), Weight2, level)
    G2 = laplacian_pyramid(np.uint32(np.uint32(img2[:, :, 1])), Weight2, level)
    B2 = laplacian_pyramid(np.uint32(np.uint32(img2[:, :, 2])), Weight2, level)

    #Fusion
    R_r = []; R_g = []; R_b = []

    for i  in range (0, len(R1)):

        #print(len(R1[i]), Weight1[i].shape)

        R_r.append(Weight1[i] * R1[i] + Weight2[i] * R2[i])
        R_g.append(Weight1[i] * G1[i] + Weight2[i] * G2[i])
        R_b.append(Weight1[i] * B1[i] + Weight2[i] * B2[i])
    
    # reconstruct 

    R = pyramid_reconstruct(R_r)
    G = pyramid_reconstruct(R_g)
    B = pyramid_reconstruct(R_b)

    fusion = cv2.merge((np.uint8(R), np.uint8(G), np.uint8(B)))

    t = time.time() - t0

    print(t)
    cv2.imshow('original', img)
    cv2.imshow('final', fusion)
    cv2.waitKey(0)


   
