import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import cv2
import math
import time
import numpy as np
from pathlib import Path
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage

from raiv_camera_calibration.perspective_calibration import PerspectiveCalibration
from raiv_libraries.src.raiv_libraries.robotUR import RobotUR
from raiv_libraries.src.raiv_libraries.simple_image_controller import SimpleImageController

MESSAGES = ("Now, put the robot tool on the point #{} and click the 'Get point' button.",
            "Now, put the robot tool on the middle of the image and click the 'Get point' button.",
            "Now, click in the image on the point #{} then click the 'Get point' button",
            "Finally, click the 'Verify' button")

#
# roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml
# rosrun usb_cam usb_cam_node >/dev/null 2>&1
#
class Calibration(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("calibration.ui",self)
        # Event handler
        self.btn_select_checkerboard_image_folder.clicked.connect(self.select_checkerboard_image_folder)
        self.btn_select_calibration_files_folder.clicked.connect(self.select_calibration_files_folder)
        self.btn_generate_calibration_files.clicked.connect(self.generate_calibration_files)
        self.btn_launch_camera_image.clicked.connect(self.launch_camera_image)
        self.btn_get_point.clicked.connect(self.get_mesure)
        # Attributs
        self.checkerboard_image_folder = None
        self.calibration_files_folder = None
        self.step = 0
        self.pixel_coord = None  # (x,y) coord of the clicked pixel
        self.robot = None

    def select_checkerboard_image_folder(self):
        dir = QFileDialog.getExistingDirectory(self, "Select checkerboard images directory")
        if dir:
            self.checkerboard_image_folder = Path(dir)
            self.lbl_checkerboard_images_folder.setText(str(self.checkerboard_image_folder))

    def select_calibration_files_folder(self):
        dir = QFileDialog.getExistingDirectory(self, "Select calibration files directory")
        if dir:
            self.calibration_files_folder = Path(dir)
            self.lbl_model_name.setText(str(self.calibration_files_folder))

    def generate_calibration_files(self):
        # Defining the dimensions of checkerboard
        checkerboard_width = int(self.edt_width.text())
        checkerboard_height = int(self.edt_height.text())
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.01)
        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []
        # Defining the world coordinates for 3D points
        objp = np.zeros((1, checkerboard_width * checkerboard_height, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checkerboard_width, 0:checkerboard_height].T.reshape(-1, 2)
        prev_img_shape = None
        # Extracting path of individual image stored in a given directory
        images = list(self.checkerboard_image_folder.glob('**/*.jpg'))
        for fname in images:
            img = cv2.imread(str(fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, (checkerboard_width,checkerboard_height),
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            """
            If desired number of corner are detected, we refine the pixel coordinates and display them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (checkerboard_width,checkerboard_height), corners2, ret)
        h, w = img.shape[:2]
        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # Saving the results
        np.save(self.calibration_files_folder / 'cam_mtx.npy', mtx)
        np.save(self.calibration_files_folder / 'dist.npy', dist)
        # Refining the camera matrix using parameters obtained by calibration
        self.new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        self.cx = self.newcam_mtx[0,2]
        self.cy = self.newcam_mtx[1,2]
        np.save(self.calibration_files_folder / 'roi.npy', roi)
        np.save(self.calibration_files_folder / 'newcam_mtx.npy', self.new_camera_mtx)
        inverse_newcam_mtx = np.linalg.inv(self.new_camera_mtx)
        np.save(self.calibration_files_folder / 'inverse_newcam_mtx.npy', mtx)
        np.save(self.calibration_files_folder / 'mtx.npy', inverse_newcam_mtx)
        np.save(self.calibration_files_folder / 'new_camera_mtx.npy', self.new_camera_mtx)
        ### UNDISTORSION ####
        dst = cv2.undistort(img, mtx, dist, None, self.new_camera_mtx)
        # Displaying the undistorted image
        qimg = self._convert_opencv_to_qimage(dst)
        self.canvas.set_image(qimg)

    def launch_camera_image(self):
        self.image_controller = SimpleImageController(image_topic='/usb_cam/image_raw')
        img, self.width, self.height = self.image_controller.get_image()
        qimage = QImage(img.tobytes("raw","RGB"), self.width, self.height, QImage.Format_RGB888)
        self.canvas.set_image(qimage)
        self.robot = RobotUR()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.canvas.set_image(image)
        #self.label.setPixmap(QPixmap.fromImage(image))

    def get_mesure(self):
        if self.step == 0: # We measure the camera
            (self.X_camera, self.Y_camera, self.Z_camera) = self._get_point()
            self.world_points = np.empty((0,3), dtype=np.float32)  # (X,Y,d*) d* is the distance from your point to the camera lens. (d* = Z for the camera center)
            self.step += 1
            self.txt_explanation.setPlainText(MESSAGES[0].format(self.step)) # Next message
        elif 1 <= self.step <= 9: # We measure the x,y,z for the 9 points
            (x, y, z) = self._get_point()
            # d = self._compute_distance(x,y,z)   Il faut mettre Z, pas D. Depuis l'article, on mesure d puis on en déduit Z. Ici, on a directement Z
            # print(x,y,d)
            # self.world_points = np.append(self.world_points, [[x, y, d]], axis=0)
            self.world_points = np.append(self.world_points, [[x, y, z]], axis=0)
            self.step += 1
            if self.step <= 9:
                self.txt_explanation.setPlainText(MESSAGES[0].format(self.step))
            else:
                self.txt_explanation.setPlainText(MESSAGES[1].format(self.step)) # Next message
        elif self.step == 10: # We measure the x,y,z for the point in the center of the scene
            (self.X_center, self.Y_center, z) = self._get_point()
            self.Z_center = self._compute_distance(self.X_center, self.Y_center, z)
            self.world_points = np.insert(self.world_points, 0, [[self.X_center, self.Y_center, self.Z_center]], axis=0)
            print(self.world_points)
            self.image_points = np.array([[self.cx, self.cy]], dtype=np.int) # [u,v] center + 9 Image points
            self.step += 1
            self.txt_explanation.setPlainText(MESSAGES[2].format(self.step-10)) # Next message
        elif 11 <= self.step <= 19: # We measure the pixel coordinates of the 9 points
            print("Pixel coord = {}".format(self.pixel_coord))
            self.image_points = np.append(self.image_points, [[self.pixel_coord[0], self.pixel_coord[1]]], axis=0)
            self.step += 1
            if self.step <= 19:
                self.txt_explanation.setPlainText(MESSAGES[2].format(self.step-10)) # Next message
            else:
                self.txt_explanation.setPlainText(MESSAGES[3])  # All points are acquired, let's start calibration
                print(self.image_points)
                self._calibrate()
                self.dPoint = PerspectiveCalibration(self.calibration_files_folder)
        else:
            xyz = self.dPoint.from_2d_to_3d(self.pixel_coord)
            print(xyz)
            x = xyz[0][0] / 100
            y = xyz[1][0] / 100
            print(x,y)
            self.robot.go_to_xyz_position(x, y, 0.2)

    def _compute_distance(self,x,y,z):
        """ Return the distance between the x,y,z point and the camera"""
        dx = x - self.X_camera
        dy = y - self.Y_camera
        dz = z - self.Z_camera
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _get_point(self):
        pose = self.robot.get_current_pose()
        print(pose)
        return (pose.position.x*100, pose.position.y*100, pose.position.z*100)  # From m to cm

    def _convert_opencv_to_qimage(self, cvImg):
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        return QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)

    def _calibrate(self):
        cam_mtx, dist, roi, newcam_mtx, inverse_newcam_mtx = self.load_parameters()
        total_points_used = 10
        # For Real World Points, calculate Z from d*
        # world_points = calculate_z_total_points(world_points, X_center, Y_center)

        # Get rotation and translation_vector from the parameters of the camera, given a set of 2D and 3D points
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.world_points, self.image_points, newcam_mtx, dist,
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)
        # if self.writeValues:
        #     print("save")
        #     self.save_parameters(rotation_vector, translation_vector, newcam_mtx)
        # # Check the accuracy now
        mean, std = self.calculate_accuracy(self.world_points, self.image_points, total_points_used)
        print("Mean:{0}".format(mean) + "Std:{0}".format(std))

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

    def load_checking_parameters(self):
        rotation_vector = np.load(self.calibration_files_folder / 'rotation_vector.npy')
        translation_vector = np.load(self.calibration_files_folder / 'translation_vector.npy')
        R_mtx = np.load(self.calibration_files_folder / 'R_mtx.npy')
        Rt = np.load(self.calibration_files_folder / 'Rt.npy')
        P_mtx = np.load(self.calibration_files_folder / 'P_mtx.npy')
        inverse_newcam_mtx = np.load(self.calibration_files_folder / 'inverse_newcam_mtx.npy')
        return rotation_vector, translation_vector, R_mtx, Rt, P_mtx, inverse_newcam_mtx

    # Load parameters from the camera
    def load_parameters(self):
        # load camera calibration
        cam_mtx = np.load(self.calibration_files_folder / 'cam_mtx.npy')
        dist = np.load(self.calibration_files_folder / 'dist.npy')
        roi = np.load(self.calibration_files_folder / 'roi.npy')
        newcam_mtx = np.load(self.calibration_files_folder / 'newcam_mtx.npy')
        inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
        np.save(self.calibration_files_folder / 'inverse_newcam_mtx.npy', inverse_newcam_mtx)
        # if self.display:
        #     print("Camera Matrix :\n {0}".format(cam_mtx))
        #     print("Dist Coeffs :\n {0}".format(dist))
        #     print("Region of Interest :\n {0}".format(roi))
        #     print("New Camera Matrix :\n {0}".format(newcam_mtx))
        #     print("Inverse New Camera Matrix :\n {0}".format(inverse_newcam_mtx))
        return cam_mtx, dist, roi, newcam_mtx, inverse_newcam_mtx

if __name__ == '__main__':
    import rospy
    rospy.init_node('explore2')
    app = QApplication(sys.argv)
    gui = Calibration()
    gui.show()
    sys.exit(app.exec_())