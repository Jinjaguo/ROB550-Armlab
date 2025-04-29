#!/usr/bin/env python3

"""!
Class to represent the camera.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError

from pdb import set_trace
from enum import IntEnum

class BlockColor(IntEnum):
    RED = 0
    ORANGE = 1
    YELLOW = 2
    GREEN = 3
    BLUE = 4
    PURPLE = 5
    ERROR = 6

    def str_to_enum(color: str):
        if color == 'red':
            return BlockColor.RED
        elif color == 'orange':
            return BlockColor.ORANGE
        elif color == 'yellow':
            return BlockColor.YELLOW
        elif color == 'green':
            return BlockColor.GREEN
        elif color == 'blue':
            return BlockColor.BLUE
        elif color == 'purple':
            return BlockColor.PURPLE
        else:
            return BlockColor.ERROR


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        self.DepthFrameRawNoHomography = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720, 1280, 3)).astype(np.uint8)

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        # self.intrinsic_matrix = np.array([[902.1877, 0, 662.3499],[0, 902.3912, 372.2278],[0,
        # self.intrinsic_matrix0, 1]])
        self.intrinsic_matrix = np.eye(3, dtype=np.float32)
        self.extrinsic_matrix = np.array([[1, 0, 0, -9], [0, -0.9866, 0.163, 225], [0, -0.163, -0.9866, 1032],
                                          [0, 0, 0, 1]])  # rotation 189.4 about x
        self.dist_coeffs = np.zeros((5, 1))
        self.H_matrix = np.eye(3)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.click_point_world = np.zeros((3, 1))
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.detections_array = np.zeros([4, 2])
        self.detection_counter = 0
        self.depth_interp_values = np.zeros((4, 1))
        self.detected_blocks = {}
        self.ef_pos = np.zeros(3)

        self.tag_locations = np.array([[-250, -25, 0],
                                       [250, -25, 0],
                                       [250, 275, 0],
                                       [-250, 275, 0]])
        scale = 0.4
        edge_size = ((1 - scale) / 2)

        self.dest_points = np.array(
            [[1280 * edge_size, 720 * (1 - edge_size)], [1280 * (1 - edge_size), 720 * (1 - edge_size)],
             [1280 * (1 - edge_size), 720 * (edge_size)], [1280 * edge_size, 720 * edge_size]])
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        self.save_detections = True
        self.block_in_gripper = False

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """

        pass

    def pixel2World(self, x, y, d):
        """!
        @brief      Convert pixel coordinates to world coordinates

        """
        pt = np.array([x, y, 1])

        if self.cameraCalibrated:
            pt_h = self.homography(np.array([x, y]))
            y_endpt2 = np.interp(pt_h[0], self.dest_points[:2, 0], self.depth_interp_values[:2, 0])
            y_endpt1 = np.interp(pt_h[0], self.dest_points[:2, 0], self.depth_interp_values[3:1:-1, 0])

            d_offset = np.interp(pt_h[1], self.dest_points[2:0:-1, 1], np.array([y_endpt1, y_endpt2]))

            d += d_offset

        k_inv = np.linalg.inv(self.intrinsic_matrix)
        camera_frame = d * k_inv @ pt  # inverse intrinsic matrix

        extrensic_inv = np.linalg.inv(self.extrinsic_matrix)

        world_frame = extrensic_inv @ np.append(camera_frame, 1)  # inverse extrinsic matrix

        return world_frame[:3]

    def world2Pixel(self, x, y, z):
        """!
        @brief      Convert world coordinates to pixel coordinates

        """
        world_frame = np.array([x, y, z, 1])

        camera_frame = self.extrinsic_matrix @ world_frame

        projected_pt = self.intrinsic_matrix @ camera_frame[:3]
        projected_pt /= camera_frame[2]

        return projected_pt[:2]

    def homography(self, pt):
        """!
        @brief Compute Homography transformation
            
            """
        pt = self.H_matrix @ np.append(pt, 1)
        return pt[:2] / pt[2]

    def inverse_homography(self, pt):
        """!
        @brief Compute inverse Homography transformation
            
            """
        pt = np.linalg.inv(self.H_matrix) @ np.append(pt, 1)
        return pt[:2] / pt[2]
    
    # def convert_2D_to_3D_world(self, x, y):
    #     self.world2Pixel()
    #     z = self.DepthFrameRawNoHomography[inv_h_pt[1]][inv_h_pt[0]]

    #     self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
    #                                     (pt[0], pt[1], z))
    #     world_frame = self.pixel2World(inv_h_pt[0], inv_h_pt[1], z)

    #     return world_frame


    def gripper_poll(self):
        return self.block_in_gripper

    def blockDetector(self):
        """!
        @brief      Detect blocks from RGB and locate them in 3D space.
        """
        # Copy the video frame
        image = self.VideoFrame.copy()


        # Define the Region of Interest (ROI) with start and end coordinates
        roi = (150, 1100, 20, 650)  # x_start, x_end, y_start, y_end
        x_start, x_end, y_start, y_end = roi
        roi_frame = image[y_start:y_end, x_start:x_end]
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        # Convert the ROI from BGR to HSV color space
        roi_frame_RGB = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

        hsv = cv2.cvtColor(roi_frame_RGB, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for different colors
        color_hsv_ranges = {
            'red': {'low': [150, 100, 80], 'high': [180, 255, 255]},
            'green': {'low': [50, 100, 60], 'high': [90, 200, 255]},
            'blue': {'low': [99, 93, 73], 'high': [105, 255, 255]},
            'yellow': {'low': [20, 180, 160], 'high': [50, 255, 255]},
            'orange': {'low': [5, 120, 157], 'high': [13, 209, 255]},
            'purple': {'low': [105, 55, 25], 'high': [200, 200, 120]}
        }

        

        # Kernel for morphological operations (erosion and dilation)
        kernel = np.ones((5, 5), np.uint8)
        self.block_contours = []
        self.block_detections = []

        # Initialize an empty dictionary to store the detected blocks
        if self.save_detections:
            self.detected_blocks = {}
        
        # Loop through each color range
        self.block_in_gripper = False

        for color, ranges in color_hsv_ranges.items():
            #set_trace()
            lower = np.array(ranges['low'])
            upper = np.array(ranges['high'])
            mask = cv2.inRange(hsv, lower, upper)

            # Remove noise using erosion and dilation
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)

            # Optionally, apply Gaussian blur to further reduce noise
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # Find contours within the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print(f"Detecting {color}: {len(contours)} contours found")

            # Process the detected contours


            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Filter out contours with a very small area
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)
                    size = "Large" if area > 1000 else "Small"

                    # Calculate the angle of the contour
                    rect = cv2.minAreaRect(contour)
                    width = rect[1][0]
                    height = rect[1][1]
                    ratio = width / height

                    flag = 0.5 < ratio < 2
                    angle = 0 if abs(rect[2]) < 0.5 or abs(rect[2] - 90) < 0.5 else rect[2]

                    if flag:
                        # Calculate the distance from the center to the origin (0, 0)
                        center_inv_h = self.inverse_homography(np.array([center_x + x_start, center_y + y_start]))
                        # if self.cameraCalibrated:
                            # set_trace()
                            # pass
                        x_world, y_world, dep_world = self.detectBlocksInDepthImage(center_inv_h[0], center_inv_h[1],
                                                                                    self.DepthFrameRawNoHomography[
                                                                                        int(center_inv_h[1]), int(center_inv_h[0])])
                        distance = np.sqrt(x_world ** 2 + y_world ** 2)
                        ef_x, ef_y, ef_z = self.ef_pos
                        # distance_ef = np.sqrt(ef_x ** 2 + ef_y ** 2)

                        if(not self.block_in_gripper) and (abs(x_world - ef_x) < 75 and abs(y_world - ef_y) < 75 and abs(dep_world - ef_z) < 75):
                            self.block_in_gripper = True

                        if self.save_detections:
                            if not self.block_in_gripper and dep_world < 300:
                                key = (distance, BlockColor.str_to_enum(color))  # Create a key as a tuple (distance, color)
                                self.detected_blocks[key] = (x_world, y_world, dep_world, np.pi/180*angle, 38 if size == "Large" else 25)

                        # Draw the bounding box and center point
                        # cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
                        corner_points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) + np.array([x_start, y_start])
                        center = (center_x + x_start, center_y + y_start)
                        rotation = cv2.getRotationMatrix2D(center, -angle, 1)
                        corner_points_rotated = cv2.transform(np.array([corner_points]), rotation)
                        cv2.polylines(image, [corner_points_rotated], 1, (255, 0, 0), 2)

                        cv2.circle(image, (center_x + x_start, center_y + y_start), 5, (0, 255, 0), -1)

                        # Add labels (color and size) to the image
                        label = f"{color.capitalize()} ({size})"
                        cv2.putText(image, label, (x+130, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # cv2.putText(image, f"Center: ({center_x}, {center_y})", (x, y + h + 10),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(image, f"Angle: {angle:.2f}deg", (x+130, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 2)

        # Store the processed frame for further use
        self.GridFrame = image

    def detectBlocksInDepthImage(self, x, y, d):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        world_frame = self.pixel2World(x, y, d)
        x, y = world_frame[:2]
        depth = world_frame[2]

        return x, y, depth

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """

        self.GridFrame = self.VideoFrame.copy()

        for x, y in zip(self.grid_points[0].ravel(), self.grid_points[1].ravel()):
            projected_pt = self.world2Pixel(x, y, 0)

            projected_pt = self.homography(projected_pt).astype(np.int32)

            cv2.circle(self.GridFrame, tuple(projected_pt), 5, (255, 0, 0), -1)

        self.blockDetector()

    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """

        modified_image = self.VideoFrame.copy()

        self.detection_counter = 0
        for detection in msg.detections:
            center = self.homography(np.array([detection.centre.x, detection.centre.y]))

            try:
                self.detections_array[detection.id - 1, :] = center
            except:
                # print("Out of bounds")
                pass

            cv2.circle(modified_image, tuple(center.astype(np.int32)), 1, (255, 0, 0), 5)
            pts = np.array([self.homography([corner.x, corner.y]) for corner in detection.corners], np.int32)
            cv2.polylines(modified_image, [pts], True, (0, 255, 0), 1)
            text = f"ID: {detection.id}"
            cv2.putText(modified_image, text, (int(center[0]), int(center[1] + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 1, cv2.LINE_AA)
            self.detection_counter += 1

        self.TagImageFrame = modified_image

    def calibrate(self):
        """!
        @brief      Calibrate the camera
        """
        # print(self.intrinsic_matrix)

        if self.cameraCalibrated is True:
            print("Camera already calibrated")
            return

        [_, R_exp, t] = cv2.solvePnP(self.tag_locations.astype(np.float32),
                                     self.detections_array.astype(np.float32),
                                     self.intrinsic_matrix.astype(np.float32),
                                     None,
                                     flags=cv2.SOLVEPNP_AP3P)
        R, _ = cv2.Rodrigues(R_exp)
        self.extrinsic_matrix = np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))

        self.H_matrix, _ = cv2.findHomography(self.detections_array, self.dest_points)

        for i in range(4):
            pt = self.pixel2World(self.detections_array[i, 0], self.detections_array[i, 1],
                                  self.DepthFrameRawNoHomography[
                                      int(self.detections_array[i, 1]), int(self.detections_array[i, 0])])
            self.depth_interp_values[i] = pt[2]
        self.cameraCalibrated = True

    def recover_homogenous_affine_transformation(self, p, p_prime):
        """points_transformed_1 = points_transformed_1 = np.dot(
        A1, np.transpose(np.column_stack((points_camera, (1, 1, 1, 1)))))np.dot(
        A1, np.transpose(np.column_stack((points_camera, (1, 1, 1, 1)))))
        Find the unique homogeneous affine transformation that
        maps a set of 3 points to another set of 3 points in 3D
        space:

            p_prime == np.dot(p, R) + t

        where `R` is an unknown rotation matrix, `t` is an unknown
        translation vector, and `p` and `p_prime` are the original
        and transformed set of points stored as row vectors:

            p       = np.array((p1,       p2,       p3))
            p_prime = np.array((p1_prime, p2_prime, p3_prime))

        The result of this function is an augmented 4-by-4
        matrix `A` that represents this affine transformation:

            np.column_stack((p_prime, (1, 1, 1))) == \
                np.dot(np.column_stack((p, (1, 1, 1))), A)

        Source: https://math.stackexchange.com/a/222170 (robjohn)
        """

        # construct intermediate matrix
        Q = p[1:] - p[0]
        Q_prime = p_prime[1:] - p_prime[0]

        # calculate rotation matrix
        R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
                   np.row_stack((Q_prime, np.cross(*Q_prime))))

        # calculate translation vector
        t = p_prime[0] - np.dot(p[0], R)

        # calculate affine transformation matrix
        return np.transpose(np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1))))


class ImageListener(Node):

    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image

        if (self.camera.cameraCalibrated):
            self.camera.VideoFrame = cv2.warpPerspective(self.camera.VideoFrame, self.camera.H_matrix, (
                self.camera.VideoFrame.shape[1], self.camera.VideoFrame.shape[0]))


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        self.camera.dist_coeffs = data.d
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2

        self.camera.DepthFrameRawNoHomography = self.camera.DepthFrameRaw.copy()
        if (self.camera.cameraCalibrated):
            self.camera.DepthFrameRaw = cv2.warpPerspective(self.camera.DepthFrameRaw, self.camera.H_matrix, (
                self.camera.DepthFrameRaw.shape[1], self.camera.DepthFrameRaw.shape[0]))

        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once()  # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                               cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass

        self.executor.shutdown()


def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
