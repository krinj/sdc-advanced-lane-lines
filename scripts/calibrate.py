from layer import Layer
import numpy as np
import cv2
import os

class Calibrate(Layer):

    cal_matrix = None
    cal_distort = None

    def __init__(self, cal_dir, cal_shape):
        self.name = "Calibrate"
        self.calibrate_camera(cal_dir, cal_shape)
        self.override_original = True

    def process(self, image_input, image_original, model, output_path):
        return self.undistort(image_input)

    def calibrate_camera(self, image_dir, chess_shape):
        print("Calibrate Camera...")
        img_points = []
        obj_points = []
        img_shape = None
        for image_path in os.listdir(image_dir):

            if ".jpg" not in image_path:
                continue

            image = cv2.imread(image_dir + image_path)
            if img_shape == None:
                img_shape = image.shape[0:2]
            new_image = self.add_calibration_points(image, chess_shape, img_points, obj_points)
            self.save_image(new_image, image_path, image_dir + "WithChessDrawn/")
            print("Calibrating: ", image_path)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)
        self.cal_matrix = mtx
        self.cal_distort = dist
        print("Camera Calibration Complete")

        # Undistort the calibration images to see how it will look
        for image_path in os.listdir(image_dir):
            if ".jpg" not in image_path:
                continue
            image = cv2.imread(image_dir + image_path)
            self.save_image(self.undistort(image), image_path, image_dir + "Undistorted/")

    def undistort(self, image):
        if self.cal_matrix == None:
            return image
        return cv2.undistort(image, self.cal_matrix, self.cal_distort, None, self.cal_matrix)

    def add_calibration_points(self, image, chess_shape, img_points, obj_points):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chess_shape, None)
        image = cv2.drawChessboardCorners(image, chess_shape, corners, ret)

        #Prepare the object point - it should be the same for all images.
        object_point = np.zeros((chess_shape[0] * chess_shape[1], 3), np.float32)
        object_point[:,:2] = np.mgrid[0:chess_shape[0],0:chess_shape[1]].T.reshape(-1,2)

        if ret == True:
            # Add the points
            img_points.append(corners)
            obj_points.append(object_point)

        return image

    def save_image(self, image, image_name, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir + image_name, image);