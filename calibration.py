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
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        np.save(self.calibration_files_folder / 'roi.npy', roi)
        np.save(self.calibration_files_folder / 'newcam_mtx.npy', new_camera_mtx)
        inverse_newcam_mtx = np.linalg.inv(new_camera_mtx)
        np.save(self.calibration_files_folder / 'inverse_newcam_mtx.npy', mtx)
        np.save(self.calibration_files_folder / 'mtx.npy', inverse_newcam_mtx)
        np.save(self.calibration_files_folder / 'new_camera_mtx.npy', new_camera_mtx)
        ### UNDISTORSION ####
        dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
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
            self.world_points = np.empty((0,3), dtype=np.float32)
            self.step += 1
            self.txt_explanation.setPlainText(MESSAGES[0].format(self.step)) # Next message
        elif 1 <= self.step <= 9: # We measure the x,y,z for the 9 points
            (x, y, z) = self._get_point()
            d = self._compute_distance(x,y,z)
            print(x,y,d)
            self.world_points = np.append(self.world_points, [[x, y, d]], axis=0)
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
            self.image_points = np.array([[int(self.width/2), int(self.height/2)]], dtype=np.int)
            self.step += 1
            self.txt_explanation.setPlainText(MESSAGES[2].format(self.step-10)) # Next message
        elif 11 <= self.step <= 19: # We measure the pixel coordinates of the 9 points
            print("Pixel coord = {}".format(self.pixel_coord))
            self.image_points = np.append(self.image_points, [[self.pixel_coord[0], self.pixel_coord[1]]], axis=0)
            self.step += 1
            if self.step < 19:
                self.txt_explanation.setPlainText(MESSAGES[2].format(self.step-10)) # Next message
            else:
                self.txt_explanation.setPlainText(MESSAGES[3])
                print(self.image_points)

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

if __name__ == '__main__':
    import rospy
    rospy.init_node('explore2')
    app = QApplication(sys.argv)
    gui = Calibration()
    gui.show()
    sys.exit(app.exec_())