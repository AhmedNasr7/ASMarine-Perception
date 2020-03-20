#! usr/bin/env python

import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge , CvBridgeError


def image_preprocessing():
    rospy.init_node('images_preprocessing',anonymous=True)
    #Set up your publisher node and define its topic:
    pub_topic_node = rospy.Publisher('Perception',Image, queue_size=10)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
       img = cv2.imread('images/tomato.jpg',1)
       gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       #convert opencv image to Ros image message:
       msg_img = CvBridge().cv2_to_imgmsg(gray)
       rospy.loginfo(msg_img)
       pub_topic_node.publish(msg_img) 
       rate.sleep()



if __name__== '__main__':
    try: 
        image_preprocessing()
    except rospy.ROSInterruptException:
        pass

