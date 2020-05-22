#! usr/bin/env python

import rospy
import std_msgs
import cv2
import numpy as np
from std_msgs.msg import Int32MultiArray

def object_coordinates_callback(msg):
    print("subscribe")
    try:
       object_coordinates = msg.data
       print('object Coordinates = ',object_coordinates)
    except e:
            print(e)

# 1.define a ros node:
rospy.init_node('object_coordinates_subscriber',anonymous=True)

# 2.set up your subscriber and define its callback:
rospy.Subscriber('Yolo_Detection',Int32MultiArray, object_coordinates_callback) 

# 3.spin until (ctrl + c) :
rospy.spin()
