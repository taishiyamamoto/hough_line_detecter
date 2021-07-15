#!/usr/bin/env python

import cv2
import numpy as np
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError
import rospy

class HoughLineDetecter:
    def __init__(self):
        self.input_image_topic = rospy.get_param('~input_image_topic','camera/rgb/image_raw')
        self.result_image_topic = rospy.get_param('~result_image_topic','result_image')
        self.binarize_image_topic = rospy.get_param('~binarize_image_topic','binarize_image')

        self.binarize_threshold = rospy.get_param('~binarize_threshold',200)

        self.hough_resolution = rospy.get_param('~hough_resolution',180)
        self.hough_threshold = rospy.get_param('~hough_threshold',5)
        self.hough_minlinelength = rospy.get_param('~hough_minlinelength',20)
        self.hough_maxlinegap = rospy.get_param('~hough_maxlinegap',5)

        self.image_sub = rospy.Subscriber(self.input_image_topic,Image, self.image_callback)
        self.image_pub = rospy.Publisher(self.result_image_topic,Image, queue_size=10)
        self.binarize_image_pub = rospy.Publisher(self.binarize_image_topic,Image,queue_size=10)

        self.bridge = CvBridge()

    def image_callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        lines = self.hough_line_detect(cv_image)
        
        if lines is not None:

            for line in lines:
                x1, y1, x2, y2 = line[0]

                red_line_img = cv2.line(cv_image, (x1,y1), (x2,y2), (0,0,255), 3)
                if 0 <= abs(np.degrees(np.arctan2(-(y2-y1), x2-x1))) and abs(np.degrees(np.arctan2(-(y2-y1), x2-x1))) <= 5:
                    rospy.loginfo("horizontal line detected")
                elif 85 <= abs(np.degrees(np.arctan2(-(y2-y1), x2-x1))) and abs(np.degrees(np.arctan2(-(y2-y1), x2-x1))) <= 90:
                    rospy.loginfo("vertical line detected")
                    
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(red_line_img, "bgr8"))
            except CvBridgeError as e:
                print(e)
        
    def hough_line_detect(self,data):
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        ret,img_thresh = cv2.threshold(gray, self.binarize_threshold, 255, cv2.THRESH_BINARY)

        try:
            self.binarize_image_pub.publish(self.bridge.cv2_to_imgmsg(img_thresh, "mono8"))
        except CvBridgeError as e:
            print(e)

        lines = cv2.HoughLinesP(img_thresh, rho=1, \
                                theta           = np.pi / self.hough_resolution, \
                                threshold       = self.hough_threshold, \
                                minLineLength   = self.hough_minlinelength, \
                                maxLineGap      = self.hough_maxlinegap)

        return lines

if __name__ == '__main__':
    rospy.init_node('hough_line_detect_node')
    hough_line_detecter = HoughLineDetecter()
    rospy.spin()