
import cv2

import numpy as np
from perspective_calibration import PerspectiveCalibration
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import moveit_commander
import rospy
import rospkg
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_commander.conversions import pose_to_list
#from robot import Robot
from raiv_libraries.robotUR import RobotUR
import time

# global variables
image_coordinates = []

# Création d'un objet de la classe PerspectiveCalibration

dPoint = PerspectiveCalibration()
dPoint.setup_camera()

# initialisation du node robotUR

myRobot = RobotUR()
rospy.init_node('robotUR')

print(myRobot.get_current_pose())

# Position Initiale du robot
pose_init = Pose()
pose_init.position.x = -0.32619530151200343
pose_init.position.y = 0.3264760799654223
pose_init.position.z = 0.15268409877277422
pose_init.orientation.x = 0.5224185254931262
pose_init.orientation.y = 0.8491397286773459
pose_init.orientation.z = -0.056266648059096896
pose_init.orientation.w = 0.05361594650423735
myRobot.go_to_pose_goal(pose_init)

def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        image_coord = [x, y]

        xyz = dPoint.from_2d_to_3d(image_coord)

        print(xyz)
        # displaying the coordinates on the Shell
        # print(x, ' ', y)
        images_coordinates = image_coordinates.append([x, y])
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(int(xyz[0][0])) + ' , ' +
        #             str(int(xyz[1][0])) + ' , ' + str(int(xyz[2][0])), (x, y), font,
        #             0.5, (255, 0, 0), 2)
        cv2.putText(img, 'X', (x, y), font,
                     0.5, (255, 0, 0), 2)
        #cv2.imshow("image", img2)

        # Calcul du point visé par le click (position et orientation)

        pose_goal = Pose()
        pose_goal.position.x = -(xyz[0][0]) / 100
        pose_goal.position.y = -(xyz[1][0]) / 100
        pose_goal.position.z = 0.22
        pose_goal.orientation.x = 0.5224185254931262
        pose_goal.orientation.y = 0.8491397286773459
        pose_goal.orientation.z = -0.056266648059096896
        pose_goal.orientation.w = 0.05361594650423735
        print(pose_goal)
        myRobot.go_to_pose_goal(pose_goal)
        print("target reached")
        time.sleep(1)


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)
nom_fenetre = "webcam"
cv2.namedWindow(nom_fenetre, cv2.WND_PROP_FULLSCREEN)
while True:
   success, img = cap.read()
   if success:
        cv2.imshow("webcam", img)
        cv2.waitKey(1)
        cv2.setMouseCallback(nom_fenetre, click_event)

# setting mouse handler for the image
# and calling the click_event() function
#cv2.setMouseCallback('image', click_event)
#cv2.setMouseCallback(nom_fenetre, click_event)


# wait for a key to be pressed to exit 
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()