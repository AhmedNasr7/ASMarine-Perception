import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from shutil import copyfile
DEBUG_FLAG = False
REMOTE_FLAG = True
gate_obs = 0
if REMOTE_FLAG:
    ## original dataset images directory 
    img_dir = '/content/data/images/'
    ## existing labels directory (handmade labels)
    labels_dir = '/content/data/labels/'
    ## where to save new images
    new_dataset_dir = '/content/images/'
    ## bootlegger and gman image (together) address to be blended with the dataset
    new_imgs = '/content/scrapped/'
    ## text file to which new labels are saved
    built_labels = '/content/blended_labels.txt'
    annots_file = '/content/annots-3-20.txt'

##locals
else:
    img_dir = '/home/abdelrahman/college/datasets/robosub_transdec_dataset/images/'
    labels_dir = '/home/abdelrahman/college/datasets/robosub_transdec_dataset/labels/'
    new_dataset_dir = '/home/abdelrahman/college/datasets/robosub_transdec_dataset/blended_scrapped_dataset/'
    new_imgs = '/home/abdelrahman/college/datasets/robosub_transdec_dataset/scrapped/'
    built_labels = '/home/abdelrahman/college/datasets/robosub_transdec_dataset/new_labels.txt'
    annots_file = '/home/abdelrahman/Desktop/annots-3-20.txt'


folders = ['hat','cash','gun']
extensions = ['jpg','png','jpeg']
bg_range = list(range(1,104)) + list(range(513,543)) + list(range(590,1018)) + list(range(1369,1524)) + list(range(1572,1605))\
                + list(range(1654,1714))
labels = {'gate': '0' , 'hat': '1' , 'bottle' : '2' , 'cash' : '3' , 'gun' : '4'}


## image is a np.array . size is height*width , take their min and act on

def resize(image,size):
    h,w,_ = image.shape
    ar = w/h
    nh = size[0]
    nw = int(nh * ar)
    if nw > size[1]:
        nw = size[1]
        nh = int( 1/ar *nw)
    return cv2.resize(image,(nw,nh),cv2.INTER_LANCZOS4)



def propose_region(im_size):
    w1 = random.randint(30,im_size[1]//2)
    h1 = random.randint(30,im_size[0]//2)
    lx1 = random.randint(0,im_size[1]-w1)
    ty1 = random.randint(0,im_size[0]-h1)
    w2,h2,lx2,ty2=0,0,0,0

    if im_size[1]-(w1+lx1) < lx1:              # left half for second image
        w2 =  random.randint(min(40,lx1//2),min(lx1,im_size[1]//2))
        lx2 = random.randint(0,lx1-w2)
    else:                                   # right half for second image
        w2 =  random.randint( min(40,(im_size[1]-lx1-w1)//2) ,  min(im_size[1]-lx1-w1,im_size[1]//2))
        lx2 = random.randint( w1+lx1   ,  im_size[1]-w2)

    if im_size[0]-(ty1+h1) < ty1 :    # upper half for second image
        h2 = random.randint(min(40,ty1//2),min(ty1,(im_size[0]*3)//5))
        ty2 = random.randint(0,ty1-h2)
    else:                           # lower half for second image
        h2 = random.randint(min(40,(im_size[0]-h1-ty1)//2),min(im_size[0]-h1-ty1,(im_size[0]*3)//5))
        ty2 = random.randint(ty1+h1 , im_size[0]-h2 )

    return [[lx1,ty1,lx1+w1,ty1+h1]  ,  [lx2,ty2,lx2+w2,ty2+h2]] 
        


def blend_two_images_in_background(bg,img1,img2,alpha=0.2):
    [roi1 , roi2] = propose_region((bg.shape[0],bg.shape[1]))
    img1 = resize(img1,(roi1[3]-roi1[1],roi1[2]-roi1[0]))
    img2 = resize(img2,(roi2[3]-roi2[1],roi2[2]-roi2[0]))

    roi_for_img1 = bg[roi1[1]:roi1[1]+img1.shape[0],roi1[0]:roi1[0]+img1.shape[1],:]
    roi_for_img2 = bg[roi2[1]:roi2[1]+img2.shape[0],roi2[0]:roi2[0]+img2.shape[1],:]
    roi_for_img1 = cv2.addWeighted(img1,alpha,roi_for_img1,1-alpha,0)
    roi_for_img2 = cv2.addWeighted(img2,alpha,roi_for_img2,1-alpha,0)

    bg[roi1[1]:roi1[1]+img1.shape[0],roi1[0]:roi1[0]+img1.shape[1],:] = roi_for_img1
    bg[roi2[1]:roi2[1]+img2.shape[0],roi2[0]:roi2[0]+img2.shape[1],:] = roi_for_img2
    
    return bg , [[roi1[0], roi1[1] , roi1[0]+img1.shape[1] , roi1[1]+img1.shape[0]], \
                 [roi2[0], roi2[1] , roi2[0]+img2.shape[1] , roi2[1]+img2.shape[0]] ]



    
label_files = glob.glob(labels_dir+'*.txt')
label_files = [ label.split('/')[-1].split('.')[0] for label in label_files ]


def load_label(bg_name):
    lines = []
    if bg_name in label_files:
        with open(labels_dir+bg_name+'.txt','r') as f:
            content = f.read().split('\n')[:-1]
            for line in content:
                if line[0] == '0':
                    lines.append(xywh2xyxy( list(map(float,line.split(' ')[1:] ) ) ) )
    return lines


def xywh2xyxy(label):

    x_left = label[0]-label[2]/2
    x_right = label[0]+label[2]/2
    y_bottom = label[1]+label[3]/2
    y_top = label[1]-label[3]/2
    return [x_left,y_top,x_right,y_bottom]

def scale_label(label,im_size):
    h,w,_ = im_size
    return list(map(int,[label[0]*w , label[1] * h , label[2]*w , label[3] * h]))

def deform_label(roi,shape):
    roi_w = roi[2]-roi[0]
    roi_h = roi[3]-roi[1]
    h,w,_ = shape
    roi = (np.array(roi) + np.array(list(map(int,[-0.05*roi_w,-0.05*roi_h,0.05*roi_w,0.05*roi_h])))).tolist()
    roi = [max(roi[0],0) , max(roi[1],1)  , min(w,roi[2]) , min(roi[3],h)]
    return roi
gl = {}
def build_dataset(img,roi,names,debug=False):
    [ roi1 , roi2 ] = roi
    roi1 = deform_label(roi1,img.shape)
    roi2 = deform_label(roi2,img.shape)
    
    global gate_obs
    [bg,img1,img2] = names
    bg = bg.split('/')[-1].split('.')[0]
    gate_lines = load_label(bg)
        
    img1 = img1.split('/')[-1].split('.')[0]
    img2 = img2.split('/')[-1].split('.')[0]
    saving_name = bg+'_'+img1+'_'+img2+'.jpg'
    if saving_name in gl.keys():
        gl[saving_name]+=1
        print(saving_name,gl[saving_name], ' in build_dataset() function ')
        return
    else:
        gl[saving_name] = 1
#    cv2.imshow('img',img)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    f = 0
    if not debug:
        cv2.imwrite(new_dataset_dir+saving_name,img)
        f = open(built_labels,'a')
    label1 = labels[img1.split('_')[0]]
    label2 = labels[img2.split('_')[0]]
    for line in gate_lines:
        line = scale_label(line,img.shape)
        gate_obs+=1
        if not debug:
            f.write(saving_name+' '+' '.join(str(n) for n in line)+' 0\n')
        else:
            print(saving_name+' '+' '.join(str(n) for n in line)+' 0\n')
    if debug:
        x=0
        print(saving_name+' '+ ' '.join(str(n+5) for n in roi1)+' '+label1)
        print(saving_name+' '+ ' '.join(str(n+5) for n in roi2)+' '+label2)
    else:
        f.write(saving_name+' '+ ' '.join(str(n) for n in roi1)+' '+label1+'\n')
        f.write(saving_name+' '+ ' '.join(str(n) for n in roi2)+' '+label2+'\n')
        f.flush()
        f.close()



def load_and_blend():
    bgs = glob.glob(img_dir+'*.jpg')
    bgs = [bg for bg in bgs if int(bg.split('/')[-1].split('.')[0]) in bg_range]
    imgs = []
    for folder in folders:
        for ex in extensions:
            imgs+=glob.glob(new_imgs+folder+'/*.'+ex)
    random.shuffle(imgs)
    random.shuffle(bgs)
    
    print('%d backgroung images , %d object images'%(len(bgs),len(imgs)))
    collect_images(bgs,5,imgs,4,DEBUG_FLAG)

    imgs = []
    print("to bottle we go")
    for ex in extensions:
        imgs += glob.glob(new_imgs+'bottle/*.'+ex)
    random.shuffle(imgs)
    print('%d background images , %d bottle images'%(len(bgs),len(imgs)))
    print('currently have %d images'%len(glob.glob(new_dataset_dir+'*.jpg')))
    collect_images(bgs,4,imgs,3,DEBUG_FLAG)
    check()


def collect_images(bgs,num_bgs,imgs,num_imgs,debug=False):
    n = 0 
    for i in range(len(imgs)):
        sampled_bgs = random.sample(bgs,num_bgs)
        for j in range(num_imgs):
            k = i+j+1
            if k >= len(imgs):
                k = random.randint(0,i-1)
            for bg in sampled_bgs:
                if n%1000 == 0:
                    print(n)
                n+=1
                img1_name = imgs[i]
                img2_name = imgs[k]

                bg_image = cv2.imread(bg)
                img1 = cv2.imread(img1_name)
                img2 = cv2.imread(img2_name)

                new_img , rois = blend_two_images_in_background(bg_image,img1,img2,random.uniform(0.15,0.3))
                build_dataset(new_img,rois,[bg,img1_name,img2_name],debug)
    check()



def check():
    for key , val in gl.items():
        if val > 1 :
            print(key,val)
    print(len(gl))

def add_gate_labels():
    copied = 0
    with open(annots_file,'r') as f:
        labels = f.read().split('\n')
    for label in labels:
        if label != '' and label!= '\n' and label[-1] == '0':
            with open(built_labels,'a') as fw:
                fw.write(label+'\n')
            if not os.path.exists(new_dataset_dir+label.split(' ')[0]):
                copyfile(img_dir+label.split(' ')[0],new_dataset_dir+label.split(' ')[0])
                copied += 1

    print('moved %d'%copied)

if __name__ == '__main__':
    load_and_blend()
    add_gate_labels()
    print('gate_obs count = %d'%gate_obs)
    

#img.shape = height * width * channels