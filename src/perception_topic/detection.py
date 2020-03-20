#! usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

def image_callback(msg):
    print("Received an image correctly!")
    try:
       #convert Ros Image message to cv2 image:
       detected_img = CvBridge().imgmsg_to_cv2(msg)
       #save your image as jpeg
       cv2.imwrite('received_img.jpeg',detected_img)
    except CvBridgeError, e:
        print(e)
     
  

def main():
    rospy.init_node('detection',anonymous=True)
    #set up your subscriber and define its callback:
    rospy.Subscriber("Perception",Image,image_callback) 
    #spin until (ctrl + c) :
    rospy.spin()

if __name__ == '__main__':
    main()
