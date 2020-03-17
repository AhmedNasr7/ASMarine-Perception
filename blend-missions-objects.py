import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random




def blend(obj, water_im, img_name, obj_name):

    #heights = [100, 200, 250] # 3 sizes
    heights = []
    for i in range(3):
        heights.append(random.randint(50, 300))

    weights = [0.4, 0.2] 

    versions_no = 6

    water_img_size = 600


    object_img = cv2.imread(obj,1)
    
    water_img = cv2.imread(water_im,1)

    h, w, _= object_img.shape

    aspect_ratio = w / h


    
    water_img = cv2.resize(water_img,(water_img_size, water_img_size))

    obj_annotations =  './obj-annotations.txt'

    


    for i in range(versions_no):

        h = heights[i // 2] # one height for each weight
        w = int(aspect_ratio * h)
        weight = weights[i // 3]

        ## generating random positions

        lower_roi_x = random.randint(0, water_img_size - 12 - w)
        upper_roi_x = lower_roi_x + w
        lower_roi_y = random.randint(0, water_img_size - 12 - h)
        upper_roi_y = lower_roi_y + h


        #print(h, w, aspect_ratio)
        object_img = cv2.resize(object_img, (w, h))

        roi = water_img[lower_roi_x:upper_roi_x, lower_roi_y:upper_roi_y, :]
        gray_obj = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_obj, 0, 150, cv2.THRESH_BINARY_INV)       ##white , black bg


        background = np.zeros((h, w, 3),dtype=np.uint8) + 170
        #print(mask.shape, background.shape)

        object_img = cv2.copyTo(background, mask) + object_img
        #print(roi.shape)
        roi = cv2.resize(roi, (w, h))

        #print(object_img.shape, roi.shape)
        
        blended_roi = cv2.addWeighted(object_img, weight ,roi ,1 - weight, 0)
        blended = water_img.copy()
        blended[lower_roi_y:upper_roi_y, lower_roi_x:upper_roi_x, :] = blended_roi
        
        
        if not os.path.isdir('./dst/'): 
            os.system('mkdir ./dst/')
        
        nimg_name = img_name + '-' + str(i) + '.jpg'
        img_pth = './dst/' + nimg_name

        cv2.imwrite(img_pth, blended)

        ## write annotations 
        labels = {'paper': 5, 
                    'phone': 6,
                    'badge': 7,
                    'bottle': 8,
                    'cash': 9,
                    'tommygun': 10}

        label = labels[obj_name]
        #line_list = [nimg_name, str(lower_roi_x - 4), str(lower_roi_y - 3), str(w + 4), str(h + 5), str(label)]
        line_list = [nimg_name, str(lower_roi_x - 7), str(lower_roi_y - 5), str(upper_roi_x + 6), str(upper_roi_y + 7), str(label)]

        line = ' '.join(line_list) + '\n'

        with open(obj_annotations, 'a') as f:
            f.write(line)
        


cwd = os.getcwd()


print(cwd)
objs_dir = os.path.join(cwd, 'objects')
imgs_dir = os.path.join(cwd, 'images')

annotations_file = './annots-3-20.txt'


lines = []
with open(annotations_file, 'r') as f:
    lines = f.readlines()


counter = 0
gcounter = 0
gate_imgs = []
bad_imgs = [str(x) + '.jpg' for x in range(104, 506)]

for line in lines:
    counter += 1
    line_list = line.split()
    #print(line_list)
    if line_list[-1] == '0':
        gcounter += 1
        gate_imgs.append(line_list[0])

print('number of imgs: ', (counter - gcounter - len(bad_imgs)) * 6 * 6)
print(gcounter)
    


img_cnt = 0


for img in os.listdir(imgs_dir):

    

    if img in gate_imgs or img in bad_imgs:
        continue
	
    img_cnt += 1


    for obj in os.listdir(objs_dir):

        img_cnt += 1


        obj_name = os.path.splitext(obj)[0]
        img_name = os.path.splitext(img)[0] + '-' + obj_name
        

        img_path = os.path.join(imgs_dir, img)
        obj_path = os.path.join(objs_dir, obj)

        blend(obj_path, water_im=img_path, img_name=img_name, obj_name=obj_name)

        print('Number of {} images blended'.format(img_cnt * 6))







