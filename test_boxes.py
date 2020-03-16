from random import randint
import cv2
import os


obj_annotations = 'obj-annotations.txt'
images_dir = './dst'


lines = []
with open(obj_annotations, 'r') as f:
    lines = f.readlines()


i = randint(0,len(lines) - 1)
print(i)
line = lines[i] # random line

line = line.split()
print(line)

img_name = line[0]
x, y, x1, y1 = int(line[1]), int(line[2]), int(line[3]), int(line[4])
label = line[5]

labels =  labels = {'paper': 5, 
                    'phone': 6,
                    'badge': 7,
                    'bottle': 8,
                    'cash': 9,
                    'tommygun': 10}



imgfile = os.path.join(images_dir, img_name)

img = cv2.imread(imgfile)

print(label)

img = cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()




