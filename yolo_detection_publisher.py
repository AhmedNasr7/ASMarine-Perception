#! usr/bin/env python
from detection import *

import numpy as np
import rospy
import sys
import std_msgs 
from std_msgs.msg import Int32MultiArray

# 1.define a ros node:
rospy.init_node('yolo_detection_publisher',anonymous=True)

# 2.Set up your publisher node: 
pub_topic_node = rospy.Publisher('Yolo_Detection',Int32MultiArray,queue_size=100)

# 3.publishing rate once every 20 milliseconds (50HZ):
rate = rospy.Rate(50)  

# 4.array for the sent pipe coardinates:
Object_Rectangle_Coordinates = Int32MultiArray()

while not rospy.is_shutdown():
     parser = argparse.ArgumentParser()
     parser.add_argument('--classes', nargs='+', type=int, help='filter by class')       
     source_img = 'images/1.jpg'
     cfg = 'cfg/yolov3-spp.cfg'
     weights = 'best-AUV-gate.weights'
     opt = parser.parse_args()
     detector = Detector(opt, source_img, cfg, weights, save_img=True)
     detections, confidence = detector.detect(0.99, 0.1)
     
     detections = np.uint32(detections)
     # the frame doesn't contain any pline:
     for i in range (len(detections)):
        # publish each detected object rectangle box coordinates:
        Object_Rectangle_Coordinates.data = detections[i]
        pub_topic_node.publish(Object_Rectangle_Coordinates)
        rospy.loginfo(Object_Rectangle_Coordinates.data)
        rate.sleep()
     #print(detections)
     #print("confidence: ", confidence)
     
