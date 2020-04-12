import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

## original dataset images directory 
img_dir = '/home/abdelrahman/college/datasets/robosub_transdec_dataset/images/'
## existing labels directory (handmade labels)
labels_dir = '/home/abdelrahman/college/datasets/robosub_transdec_dataset/labels/'
## where to save new images
new_dataset_dir = '/home/abdelrahman/college/datasets/robosub_transdec_dataset/generated/'
## bootlegger and gman image (together) address to be blended with the dataset
gate_imgs = '/home/abdelrahman/college/datasets/robosub_transdec_dataset/gate_imgs_black_sep.jpg'
## text file to which new labels are saved
built_labels = '/home/abdelrahman/college/datasets/robosub_transdec_dataset/built_labels.txt'


def xywh2xyxy(labels):
    new = []
    for label in labels:
        x_left = label[0]-label[2]/2
        x_right = label[0]+label[2]/2
        y_bottom = label[1]+label[3]/2
        y_top = label[1]-label[3]/2
        new.append([x_left,y_top,x_right,y_bottom])
    return new


def roi_coords(gate_label,img):
    im_h ,im_w = img.shape[0],img.shape[1]
    ill_ratio = False
    ## sort by left box first
    gate_label.sort(key= lambda x :x[0])
    gate_label = xywh2xyxy(gate_label)        
    roi_w = int(im_w * (gate_label[1][0]-gate_label[0][2]))
    roi_h = int (im_h * (min(gate_label[0][3],gate_label[1][3]) - max(gate_label[0][1],gate_label[1][1])))
    if roi_h/im_h < 0.75 * (gate_label[0][3]-gate_label[0][1]) or roi_h/im_h < 0.75 * (gate_label[1][3]-gate_label[1][1]):
        roi_h = int(0.8*im_h *max((gate_label[0][3]-gate_label[0][1]),(gate_label[1][3]-gate_label[1][1]) ) )
        ill_ratio = True
    if roi_w/im_w < 0.75 * (gate_label[0][2]-gate_label[0][0]) or roi_w/im_h < 0.75 * (gate_label[1][2]-gate_label[1][0]):
        roi_w = int(0.8 * im_w *max((gate_label[0][2]-gate_label[0][0]), (gate_label[1][2]-gate_label[1][0]) ) )
    roi_X_top_left = int( im_w * gate_label[0][2])
    roi_Y_top_left = int( im_h * max(gate_label[0][1],gate_label[1][1]))
    if(ill_ratio):
        roi_Y_top_left = int( im_h * min(gate_label[0][1],gate_label[1][1]))

    return [roi_X_top_left,roi_Y_top_left,roi_w,roi_h]

def build_dataset(img,roi,name,saving_name='g.jpg'):
    cv2.imwrite(new_dataset_dir+saving_name,img)
    f = open(built_labels,'a')
    f.write(saving_name+' '+str(int(roi[0]-6))+' '+str(int(roi[1]-6))+' '+str(int(roi[0]+roi[2]/2))+' '+str(int(roi[1]+roi[3]+6))+' 2\n')
    f.write(saving_name+' '+str(int(roi[0]+roi[2]/2))+' '+str(int(roi[1]-6))+' '+str(int(roi[0]+roi[2]+6))+' '+str(int(roi[1]+roi[3]+6))+' 3\n')
    f.flush()
    f.close()

def blend(roi,backbone,external,img_name,alpha=0.4,saving_name='g.jpg'):
    external = cv2.resize(external,(roi[2],roi[3]))
    roi_img = backbone[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2],:]
    roi_img = cv2.addWeighted(external,alpha,roi_img,1-alpha,0)
    
    backbone[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2],:] = roi_img
    build_dataset(backbone,roi,img_name,saving_name)

def copy_gate(img_name,lines,dims):
    new_label= []
    f = open(built_labels,'a')
    for line in lines:
        if line[0]==0:
            tempo = line[1:]
            new_label.append(xywh2xyxy([line[1:]]))
            new_label[-1] = [str(int(new_label[-1][0][0]*dims[1])) ,\
                 str(int(new_label[-1][0][1]*dims[0])) ,str(int(new_label[-1][0][2]*dims[1])) ,str(int(new_label[-1][0][3]*dims[0])) ]
            data =img_name+' '.join(new_label[-1])+' 0\n'
            f.write(data)
            f.flush()
            
    f.close()


def find_obj(lines,img,img_name):
    gate = []
    buoy = []
    lines = [list(map(float,line.split())) for line in lines]
    lines.pop()

    for line in lines:
        if line[0]==0:
            gate.append(line[1:])
        elif line[0] == 1:
            buoy.append(line[1:])
    if(len(gate)>=2):
        roi = roi_coords(gate,img)
        for i in range(2):
            blend(roi,img,cv2.imread(gate_imgs,1),img_name,0.18+0.06*i+(random.uniform(0-0.02*i,0.08)),saving_name=img_name+('g'*(i+1))+'.jpg')
            copy_gate(img_name+('g'*(i+1))+'.jpg ',lines,img.shape)
    elif(len(gate)==0):
        im_h ,im_w = img.shape[0],img.shape[1]
        m = min(im_h,im_w)
        for i in range(3):
            for j in range(2):
                roi = [random.uniform(0+0.15*i,1-0.2*i)*m ,random.uniform(0+0.15*i,1-0.2*i)*m ]
                while ((m - max(roi[0],roi[1])) < 120):
                    roi = [random.uniform(0+0.15*i,1-0.2*i)*m ,random.uniform(0+0.15*i,1-0.2*i)*m ]
                width = (random.uniform(random.uniform(0.4+0.15*i,1),1) * (m - max(roi[0],roi[1])))
                roi.append(width)
                roi.append(width / (2+(random.uniform(0,0.4)-0.2)))
                roi = list(map(int,roi))
                blend(roi,img.copy(),cv2.imread(gate_imgs,1),img_name,0.18+0.06*j+(random.uniform(0-0.02*j,0.08)),saving_name=img_name+('g'*(j+1))+str(i+1)+'.jpg')
            


def concatenate_imgs(img1,img2):
    return np.concatenate((img1,np.full((273,6,3),fill_value=255,dtype=np.uint8),img2),axis=1)
    

splitter = lambda labels : img_dir+labels.split('/')[-1].split('.')[0]+'.jpg'    

def load_images_and_labels():
    labels = glob.glob(labels_dir+'*.txt')
    imgs = list(map(splitter,labels))

    for img_name , label in zip(imgs,labels):
        img = cv2.imread(img_name,1)
#        cv2.imshow('img',img)
        f = open(label,'r')
        lines = f.read().split('\n')
        if int(img_name.split('/')[-1].split('.')[0]) >= 104 and int(img_name.split('/')[-1].split('.')[0]) <= 505 :
            continue
        find_obj(lines,img,img_name.split('/')[-1].split('.')[0])
        f.close()

#        k = cv2.waitKey() & 0xFF
#        if k == ord('e'):
#            break
    cv2.destroyAllWindows()


def custom_blend(backbone,external,img_name,alpha=0.4):
    external = cv2.resize(external,(400,300))
    roi_img = backbone[50:350,50:450,:]
    roi_img = cv2.addWeighted(external,0.2,roi_img,0.8,0)
    
    backbone[50:350,50:450,:] = roi_img
    cv2.imwrite(img_name+'.png',backbone)
    

if __name__ == "__main__":
#    load_images_and_labels()
    imgs = glob.glob('/home/abdelrahman/Desktop/poll_images/*.png')
    for img in imgs:
        custom_blend(cv2.imread('/home/abdelrahman/college/datasets/robosub_transdec_dataset/images/542.jpg'),\
            cv2.imread(img),img.split('.')[0])



