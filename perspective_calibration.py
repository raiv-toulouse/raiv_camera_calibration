#!/usr/bin/env python
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


class PerspectiveCalibration:
    def __init__(self, draw=True,display = False,uwriteValues = True):
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.savedir = os.path.join(current_path, 'camera_data/')
        self.draw = draw
        self.display = display
        self.writeValues = uwriteValues

    # Get the center of the image cx and cy
    def get_image_center(self):
        # load camera calibration
        newcam_mtx = np.load(self.savedir + '/newcam_mtx.npy')

        # load center points from New Camera matrix
        cx = newcam_mtx[0, 2]
        cy = newcam_mtx[1, 2]
        fx = newcam_mtx[0, 0]
        return cx, cy

    # Load parameters from the camera
    def load_parameters(self):
        # load camera calibration
        cam_mtx = np.load(self.savedir + '/cam_mtx.npy')
        dist = np.load(self.savedir + '/dist.npy')
        roi = np.load(self.savedir + '/roi.npy')
        newcam_mtx = np.load(self.savedir + '/newcam_mtx.npy')
        inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
        np.save(self.savedir + 'inverse_newcam_mtx.npy', inverse_newcam_mtx)

        if self.display:
            print("Camera Matrix :\n {0}".format(cam_mtx))
            print("Dist Coeffs :\n {0}".format(dist))
            print("Region of Interest :\n {0}".format(roi))
            print("New Camera Matrix :\n {0}".format(newcam_mtx))
            print("Inverse New Camera Matrix :\n {0}".format(inverse_newcam_mtx))

        return cam_mtx, dist, roi, newcam_mtx, inverse_newcam_mtx

    def save_parameters(self, rotation_vector, translation_vector, newcam_mtx):
        # Rotation Vector
        np.save(self.savedir + 'rotation_vector.npy', rotation_vector)
        np.save(self.savedir + 'translation_vector.npy', translation_vector)

        # Rodrigues
        # print("R - rodrigues vecs")
        R_mtx, jac = cv2.Rodrigues(rotation_vector)
        print(R_mtx)
        print("r_mtx created")
        np.save(self.savedir + 'R_mtx.npy', R_mtx)
        print("r_mtx saved")
        # Extrinsic Matrix
        Rt = np.column_stack((R_mtx, translation_vector))
        # print("R|t - Extrinsic Matrix:\n {0}".format(Rt))
        np.save(self.savedir + 'Rt.npy', Rt)

        # Projection Matrix
        P_mtx = newcam_mtx.dot(Rt)
        # print("newCamMtx*R|t - Projection Matrix:\n {0}".format(P_mtx))
        np.save(self.savedir + 'P_mtx.npy', P_mtx)

    def load_checking_parameters(self):
        rotation_vector = np.load(self.savedir + '/rotation_vector.npy')
        translation_vector = np.load(self.savedir + '/translation_vector.npy')
        R_mtx = np.load(self.savedir + '/R_mtx.npy')
        Rt = np.load(self.savedir + '/Rt.npy')
        P_mtx = np.load(self.savedir + '/P_mtx.npy')
        inverse_newcam_mtx = np.load(self.savedir + '/inverse_newcam_mtx.npy')

        return rotation_vector, translation_vector, R_mtx, Rt, P_mtx, inverse_newcam_mtx

    # Calculate the real Z coordinate based on the center of the images
    @staticmethod
    def _calculate_z_total_points(world_points, X_center, Y_center):
        total_points_used = len(world_points)

        for i in range(1, total_points_used):
            # start from 1, given for center Z=d*
            # to center of camera
            wX = world_points[i, 0] - X_center
            wY = world_points[i, 1] - Y_center
            wd = world_points[i, 2]

            d1 = np.sqrt(np.square(wX) + np.square(wY))
            wZ = np.sqrt(np.square(wd) - np.square(d1))
            world_points[i, 2] = wZ

        return world_points

    # Lets the check the accuracy here :
    # In this script we make sure that the difference and the error are acceptable in our project.
    # If not, maybe we need more calibration images and get more points or better points
    def calculate_accuracy(self, worldPoints, imagePoints, total_points_used):
        s_arr = np.array([0], dtype=np.float32)
        size_points = len(worldPoints)
        s_describe = np.empty((size_points,), dtype=np.float32)

        rotation_vector, translation_vector, R_mtx, Rt, P_mtx, inverse_newcam_mtx = self.load_checking_parameters()

        for i in range(0, size_points):
            print("=======POINT # " + str(i) + " =========================")

            print("Forward: From World Points, Find Image Pixel\n")
            XYZ1 = np.array([[worldPoints[i, 0], worldPoints[i, 1], worldPoints[i, 2], 1]], dtype=np.float32)
            XYZ1 = XYZ1.T
            print("---- XYZ1\n")
            print(XYZ1)
            suv1 = P_mtx.dot(XYZ1)
            print("---- suv1\n")
            print(suv1)
            s = suv1[2, 0]
            uv1 = suv1 / s
            print("====>> uv1 - Image Points\n")
            print(uv1)
            print("=====>> s - Scaling Factor\n")
            print(s)
            s_arr = np.array([s / total_points_used + s_arr[0]], dtype=np.float32)
            s_describe[i] = s
            if self.writeValues:
                np.save(self.savedir + 's_arr.npy', s_arr)

            print("Solve: From Image Pixels, find World Points")

            uv_1 = np.array([[imagePoints[i, 0], imagePoints[i, 1], 1]], dtype=np.float32)
            uv_1 = uv_1.T
            print("=====> uv1\n")
            print(uv_1)
            suv_1 = s * uv_1
            print("---- suv1\n")
            print(suv_1)

            print("Get camera coordinates, multiply by inverse Camera Matrix, subtract tvec1\n")
            xyz_c = inverse_newcam_mtx.dot(suv_1)
            xyz_c = xyz_c - translation_vector
            print("---- xyz_c\n")
            inverse_R_mtx = np.linalg.inv(R_mtx)
            XYZ = inverse_R_mtx.dot(xyz_c)
            print("---- XYZ\n")
            print(XYZ)

        s_mean, s_std = np.mean(s_describe), np.std(s_describe)

        print(">>>>>>>>>>>>>>>>>>>>> S RESULTS\n")
        print("Mean: " + str(s_mean))
        # print("Average: " + str(s_arr[0]))
        print("Std: " + str(s_std))

        print(">>>>>> S Error by Point\n")

        for i in range(0, total_points_used):
            print("Point " + str(i))
            print("S: " + str(s_describe[i]) + " Mean: " + str(s_mean) + " Error: " + str(s_describe[i] - s_mean))

        return s_mean, s_std

    def from_3d_to_2d(self, image, world_coordinates, draw=False):
        # load camera calibration
        dist = np.load(self.savedir + 'dist.npy')
        newcam_mtx = np.load(self.savedir + 'newcam_mtx.npy')
        rotation_vector = np.load(self.savedir + 'rotation_vector.npy')
        translation_vector = np.load(self.savedir + 'translation_vector.npy')

        # Expected this format -> np.array([(0.0, 0.0, 30)])
        world_coordinates = np.array([world_coordinates])
        (new_point2D, jacobian) = cv2.projectPoints(world_coordinates, rotation_vector, translation_vector, newcam_mtx,
                                                    dist)
        if draw or self.draw:
            cv2.circle(image, (int(new_point2D[0][0][0]), int(new_point2D[0][0][1])), 5, (255, 0, 0), -1)
            # Display image
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()

        return new_point2D

    def from_2d_to_3d(self, image_coordinates):
        # load camera calibration
        cam_mtx, dist, roi, newcam_mtx, inverse_newcam_mtx = self.load_parameters()
        R_mtx = np.load(self.savedir + 'R_mtx.npy')
        inverse_R_mtx = np.linalg.inv(R_mtx)
        s_arr = np.load(self.savedir + 's_arr.npy')
        translation_vector = np.load(self.savedir + 'translation_vector.npy')
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

    def setup_camera(self):
        cam_mtx, dist, roi, newcam_mtx, inverse_newcam_mtx = self.load_parameters()
        # load center points from New Camera matrix
        cx, cy = self.get_image_center()
        print(cx, cy)
        total_points_used = 10

        X_center = 26.71
        Y_center = -1.71
        Z_center = 48.75
        # COORDINATES OF REAL ENVIRONMENT
        world_points = np.array([[X_center, Y_center, Z_center],
                                 [36.96, 12.21, 48.93],
                                 [37.29, 0.01, 48.66],
                                 [37.49, -11.85, 51.27],
                                 [30.94, 11.98, 48.55],
                                 [31.15, 0.02, 48.31],
                                 [31.06, -11.79, 50.89],
                                 [24.53, 12.12, 49.07],
                                 [24.39, 0.01, 48.82],
                                 [24.46, -11.79, 51.37]], dtype=np.float32)

        # MANUALLY INPUT THE DETECTED IMAGE COORDINATES HERE - Using function onclick
        # [u,v] center + 9 Image points
        image_points = np.array([[640, 480],   # Coordinates of center point (middle of the image)
                                 [282, 226],
                                 [603, 223],
                                 [903, 221],
                                 [289, 395],
                                 [603, 389],
                                 [905, 383],
                                 [297, 563],
                                 [608, 553],
                                 [906, 542]
                                 ], dtype=np.float32)
        # For Real World Points, calculate Z from d*
        # world_points = calculate_z_total_points(world_points, X_center, Y_center)

        # Get rotation and translation_vector from the parameters of the camera, given a set of 2D and 3D points
        (success, rotation_vector, translation_vector) = cv2.solvePnP(world_points, image_points, newcam_mtx, dist,
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)

        if self.writeValues:
            print("save")
            self.save_parameters(rotation_vector, translation_vector, newcam_mtx)

        # # Check the accuracy now
        mean, std = self.calculate_accuracy(world_points, image_points, total_points_used)
        print("Mean:{0}".format(mean) + "Std:{0}".format(std))

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
