#!/usr/bin/env python

import numpy as np
from pathlib import Path

#
# Perform a conversion from 2D pixel to XYZ 3D coordinates expressed in the robot frame.
# Use it AFTER a camera calibration (which provides the necessary files : newcam_mtx.npy, R_mtx.npy, S_arr.npy and translation_vector.npy)
#
class PerspectiveCalibration:
    def __init__(self, savedir):
        self.savedir = Path(savedir)
        # load camera calibration
        newcam_mtx = np.load(self.savedir / 'newcam_mtx.npy')
        self.inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
        R_mtx = np.load(self.savedir / 'R_mtx.npy')
        self.inverse_R_mtx = np.linalg.inv(R_mtx)
        s_arr = np.load(self.savedir / 's_arr.npy')
        self.translation_vector = np.load(self.savedir / 'translation_vector.npy')
        self.scalingfactor = s_arr[0]

    def from_2d_to_3d(self, image_coordinates):
        # Expected this format -> np.array([(0.0, 0.0, 30)])
        u, v = image_coordinates
        # Solve: From Image Pixels, find World Points
        uv_1 = np.array([[u, v, 1]], dtype=np.float32)
        uv_1 = uv_1.T
        suv_1 = self.scalingfactor * uv_1
        xyz_c = self.inverse_newcam_mtx.dot(suv_1)
        xyz_c = xyz_c - self.translation_vector
        XYZ = self.inverse_R_mtx.dot(xyz_c)
        return XYZ


if __name__ == '__main__':
    object = PerspectiveCalibration('../../camera_data')
    image_coordinates = [354.0, 207.0]
    new_point3D = object.from_2d_to_3d(image_coordinates)
    print(new_point3D)
