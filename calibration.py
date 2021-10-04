import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import cv2
import numpy as np
import glob
from pathlib import Path


class Calibration(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("calibration.ui",self)
        # Event handler
        self.btn_select_checkerboard_image_folder.clicked.connect(self.select_checkerboard_image_folder)
        self.btn_select_calibration_files_folder.clicked.connect(self.select_calibration_files_folder)
        self.btn_generate_calibration_files.clicked.connect(self.generate_calibration_files)
        # Attributs
        self.checkerboard_image_folder = None
        self.calibration_files_folder = None

    def select_checkerboard_image_folder(self):
        dir = QFileDialog.getExistingDirectory(self, "Select checkerboard images directory")
        if dir:
            self.checkerboard_image_folder = Path(dir)

    def select_calibration_files_folder(self):
        dir = QFileDialog.getExistingDirectory(self, "Select calibration files directory")
        if dir:
            self.calibration_files_folder = Path(dir)

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
        images = glob.glob(self.checkerboard_image_folder/'*.jpg')
        for fname in images:
            img = cv2.imread(fname)
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
            cv2.imshow('img', img)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()
        h, w = img.shape[:2]
        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # Printing  and saving the results
        print("Camera matrix : \n")
        print(mtx)
        np.save(self.calibration_files_folder / 'cam_mtx.npy', mtx)
        print("Dist : \n")
        print(dist)
        np.save(self.calibration_files_folder / 'dist.npy', dist)
        ### UNDISTORSION ####
        # Refining the camera matrix using parameters obtained by calibration
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        print("Region of Interest: \n")
        print(roi)
        np.save(self.calibration_files_folder / 'roi.npy', roi)
        print("New Camera Matrix: \n")
        # print(newcam_mtx)
        np.save(self.calibration_files_folder / 'newcam_mtx.npy', new_camera_mtx)
        print(np.load(self.calibration_files_folder / 'newcam_mtx.npy'))
        inverse_newcam_mtx = np.linalg.inv(new_camera_mtx)
        print("Inverse New Camera Matrix: \n")
        print(inverse_newcam_mtx)
        np.save(self.calibration_files_folder / 'inverse_newcam_mtx.npy', mtx)
        np.save(self.calibration_files_folder / 'mtx.npy', inverse_newcam_mtx)
        np.save(self.calibration_files_folder / 'new_camera_mtx.npy', new_camera_mtx)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = Calibration()
    gui.show()
    sys.exit(app.exec_())