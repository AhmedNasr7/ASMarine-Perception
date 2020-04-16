import numpy as np
import cv2
import math
from google.colab.patches import cv2_imshow
img = cv2.imread('/content/140.png') 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur,50,150,apertureSize = 3) 
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=10)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
cv2_imshow(img)
print(lines)
count =0
for x1,y1,x2,y2 in lines[:,0]:
  if ((count < 2) and (count == 1)):
    m2=(y2-y1)/(x2-x1)
    print ("here m2",m2)
  elif (count <1):
    m1=(y2-y1)/(x2-x1)
    count=count+1
    print ("here m1",m1)
N= m2-m1
#print(N)
D=1+m2*m1
#print(N/D)
if D==0:
  angle = 90
elif N/D <0: 
  tanthetha=np.abs(N/D)
  angle=180 - math.degrees(math.atan(tanthetha))
else:
  tanthetha=np.abs(N/D)
  angle=math.degrees(math.atan(tanthetha))

print(angle) #error in 90 as undefined by dividing by 0
             # and 120 angle gives as 58!


