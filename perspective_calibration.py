#!/usr/bin/env python
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


class PerspectiveCalibration:
    def __init__(self, savedir):
        self.savedir = savedir


    def from_2d_to_3d(self, image_coordinates):
        # load camera calibration
        cam_mtx, dist, roi, newcam_mtx, inverse_newcam_mtx = self.load_parameters()
        R_mtx = np.load(self.savedir / 'R_mtx.npy')
        inverse_R_mtx = np.linalg.inv(R_mtx)
        s_arr = np.load(self.savedir / 's_arr.npy')
        translation_vector = np.load(self.savedir / 'translation_vector.npy')
        scalingfactor = s_arr[0]
        # Expected this format -> np.array([(0.0, 0.0, 30)])
        u, v = image_coordinates
        # Solve: From Image Pixels, find World Points
        uv_1 = np.array([[u, v, 1]], dtype=np.float32)
        uv_1 = uv_1.T
        suv_1 = scalingfactor * uv_1
        xyz_c = inverse_newcam_mtx.dot(suv_1)
        xyz_c = xyz_c - translation_vector
        XYZ = inverse_R_mtx.dot(xyz_c)
        return XYZ

    # =====================================+++++++>  A VIRER, remplacer par une classe contenant tous les paramètres et matrices, type fichier Pickle
    def load_parameters(self):
        # load camera calibration
        cam_mtx = np.load(self.savedir / 'cam_mtx.npy')
        dist = np.load(self.savedir / 'dist.npy')
        roi = np.load(self.savedir / 'roi.npy')
        newcam_mtx = np.load(self.savedir / 'newcam_mtx.npy')
        inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
        np.save(self.savedir / 'inverse_newcam_mtx.npy', inverse_newcam_mtx)
        # if self.display:
        #     print("Camera Matrix :\n {0}".format(cam_mtx))
        #     print("Dist Coeffs :\n {0}".format(dist))
        #     print("Region of Interest :\n {0}".format(roi))
        #     print("New Camera Matrix :\n {0}".format(newcam_mtx))
        #     print("Inverse New Camera Matrix :\n {0}".format(inverse_newcam_mtx))
        return cam_mtx, dist, roi, newcam_mtx, inverse_newcam_mtx


if __name__ == '__main__':
    object = PerspectiveCalibration()
    camera = object.setup_camera()
    draw = True
    image_path = 'Calibration_allimages/webcam/loin/2021-05-04-164701.jpg'
    # world_coordinate = (17.51,17.83,-84.253)
    world_coordinate = (10.0, 22.0, 0.0)
    #new_point2D = object.from_3d_to_2d(image_path, world_coordinate, draw)

    # image_coordinates = [946.65573404,517.46556152]
    image_coordinates = [190.0, 373.0]
    new_point3D = object.from_2d_to_3d(image_coordinates)
    print(new_point3D)
