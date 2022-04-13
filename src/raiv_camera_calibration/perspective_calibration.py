#!/usr/bin/env python

import sys
import cv2
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge
import math
from shapely.geometry import LineString
from shapely.geometry import Point


#
# Perform a conversion from 2D pixel to XYZ 3D coordinates expressed in the robot frame.
# Use it AFTER a camera calibration (which provides the necessary files : newcam_mtx.npy, R_mtx.npy, S_arr.npy and translation_vector.npy)
#
class PerspectiveCalibration:
    def __init__(self, savedir, depth_image=None):

        if depth_image is None:
            depth_image = rospy.wait_for_message('/Distance_Here', Image)
            bridge = CvBridge()
            self.depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding = 'passthrough')
        else:
            self.depth_image = depth_image

        #Calculate the histogram of the depth image to get the distance value of the table by getting the most recurrent value in the image
        self.histogram = cv2.calcHist([self.depth_image], [0], None, [1000], [1,1000])
        self.background_index = self.histogram.argmax()

        # load camera calibration
        self.savedir = Path(savedir)
        newcam_mtx = np.load(self.savedir / 'newcam_mtx.npy')
        self.inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
        R_mtx = np.load(self.savedir / 'R_mtx.npy')
        self.inverse_R_mtx = np.linalg.inv(R_mtx)
        s_arr = np.load(self.savedir / 's_arr.npy')
        self.translation_vector = np.load(self.savedir / 'translation_vector.npy')
        self.scalingfactor = s_arr[0]

    def from_2d_to_3d(self, image_coordinates):
        bridge = CvBridge()

        # Expected this format -> np.array([(0.0, 0.0, 30)])
        u, v = image_coordinates

        # Solve: From Image Pixels, find World Points
        uv_1 = np.array([[u, v, 1]], dtype=np.float32)
        uv_1 = uv_1.T
        suv_1 = self.scalingfactor * uv_1
        xyz_c = self.inverse_newcam_mtx.dot(suv_1)
        xyz_c = xyz_c - self.translation_vector
        XYZ = self.inverse_R_mtx.dot(xyz_c)

        uv_10 = np.array([[320, 240, 1]], dtype=np.float32)
        uv_10 = uv_10.T
        suv_10 = self.scalingfactor * uv_10
        xyz_c0 = self.inverse_newcam_mtx.dot(suv_10)
        xyz_c0 = xyz_c0 - self.translation_vector
        XYZ0 = self.inverse_R_mtx.dot(xyz_c0)

        #self.depth_image = rospy.wait_for_message('/Distance_Here', Image)
        #self.depth_image = bridge.imgmsg_to_cv2(self.depth_image, desired_encoding = 'passthrough')

        #check if the value of depth is coherent with the values of heihgt we are waiting for the table, i.e between the distance of the table + 3 mm and 12 centimeters high from the table
        #if not self.background_index + 3 > self.depth_image[v][u] > self.background_index - 120:
        #    XYZ = [['a'],['b'],['c']]
        #    print('Value of depth is not coherent')
        #    return XYZ

        #This value correspond to the value of the table
        #print('self background index', self.background_index)

        #This value correspond to the value of distance of the pixel selected with the coordinates (v,u)
        #print('Depth value of the selected pixel : ', self.depth_image[v][u])

        #print('coordonnée u', u)
        #print('coordonnée v', v)

        #The height of the object of the selected pixel is equal to the height of the table minus the height of the pixel
        h_object = (self.background_index - self.depth_image[v][u])/10


        #print("Height of the object : ", h_object)

        #b_prime is the coordinates of the center of the image (in robot coordinates)
        b_prime = XYZ0

        #a_prime is the coordinates of the selected pixel(in robot coordinates)
        a_prime = XYZ

        #we calculate the distance between a_prime and b_prime(in robot coordinates)
        a_prime_b_prime = math.sqrt((a_prime[0] - b_prime[0]) ** 2 + (a_prime[1] - b_prime[1]) ** 2)

        #b_prime_c is the distance between the camera and the point at the center of the image
        b_prime_c = self.background_index /10

        #print('XYZ : ', XYZ)
        #print('A prime B prime : ', a_prime_b_prime)
        #print('b_prime_c : ', b_prime_c)


       #we use the Thales Theorem to calculate the correction necessary
        correction = abs(((h_object * a_prime_b_prime) / b_prime_c))
        print(f'The correction equals {correction} cm')

        if correction > 2.5:
            print (h_object)
            print(a_prime_b_prime)
            print(b_prime_c)

        if correction != 0 :
            #We draw a circle with the diameter of the correction with the selected pixel as the center then we draw a line from the pixel to the center of the image
            #the intersection between those two object is the point we were really aiming at the beginning
            point_1 = Point(a_prime[0], a_prime[1])
            circle = point_1.buffer(correction).boundary
            line = LineString([(a_prime[0],a_prime[1]), (b_prime[0],b_prime[1])])
            intersection = circle.intersection(line)

            try:
                return intersection.coords[0][0]/100, intersection.coords[0][1]/100, h_object/100
            except Exception as e:
                print(f'An error occured : {e}')
                print(image_coordinates)
                return self.from_2d_to_3d(self, image_coordinates)

        else :
            print('aucune correction nécessaire')
            #The new coordinates of the aimed point in the robot coordinates are declared to be the coords of the intersection
            try:
                return XYZ[0]/100, XYZ[1]/100, h_object/100
            except Exception as e:
                print(f'An error occured : {e}')
                print(image_coordinates)
                return self.from_2d_to_3d(self, image_coordinates)


if __name__ == '__main__':
    object = PerspectiveCalibration('/common/calibration/camera/camera_data')
    image_coordinates = [354.0, 207.0]
    x, y, z = object.from_2d_to_3d(image_coordinates)
    print(x, y, z)
