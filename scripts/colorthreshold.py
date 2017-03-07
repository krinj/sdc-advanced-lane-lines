from layer import Layer
import numpy as np
import cv2

class ColorThreshold(Layer):

    binary_value = 255
    th_sat = (125, 255)
    th_lightness = (50, 255)
    th_sobel_x = (35, 255)
    th_sobel_y = (10, 255)
    sobel_ksize = 7

    th_sobel_dir = (0, 0.5)
    th_sobel_mag = (50, 255)
    sobel_mag_ksize = 15
    sobel_dir_ksize = 9

    def __init__(self):
        self.name = "ColorThreshold"

    def process(self, image_input, image_original, model, output_path):
        i = []

        # 0: Saturation
        i.append(self.apply_hls_threshold(image_input, 125, 255, 2))

        # 1: Lightness
        i.append(self.apply_hls_threshold(image_input, 50, 255, 1))

        # 2: X Sobel
        i.append(self.apply_sobel(image_input, 'x', 25, 255, 9))

        # 3: Y Sobel
        i.append(self.apply_sobel(image_input, 'y', 35, 255, 5))

        # 4: M Sobel
        i.append(self.apply_mag_sobel(image_input, 15, 50, 255))

        # 5: D Sobel
        i.append(self.apply_dir_sobel(image_input, 5, 0, 0.01))

        # 6: Color Ratio Filter
        i.append(self.apply_brightness_filter(image_input))

        combined = np.zeros_like(i[0])
        combined[(
            ((i[1] > 0) | (i[0] > 0)) | ((i[2] > 0) & (i[3] > 0)))] = self.binary_value

        com1 = np.zeros_like(i[0])
        com2 = np.zeros_like(i[0])
        com3 = np.zeros_like(i[0])
        com1[((i[0] > 0) & (i[1] > 0))] = self.binary_value
        com2[((i[2] > 0) & (i[6] > 0))] = self.binary_value
        com3[((com1 > 0) | (com2 > 0))] = self.binary_value

        return com3

    def apply_brightness_filter(self, image):
        b_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        r_channel = image[:, :, 2]
        r_b_ratio = r_channel / b_channel
        binary_output = np.zeros_like(b_channel)
        binary_output[((r_b_ratio > 1.15) | ((r_channel > 175) & (g_channel > 100)))] = self.binary_value
        return binary_output

    def apply_hls_threshold(self, image, t_low, t_high, channel):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, channel]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > t_low) & (s_channel <= t_high)] = self.binary_value
        return binary_output

    def apply_sobel(self, image, orient='x', t_low=0, t_high=255, kernel=3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel))
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= t_low) & (scaled_sobel <= t_high)] = self.binary_value
        return binary_output

    def apply_mag_sobel(self, image, kernel=3, t_low=0, t_high=np.pi/2):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Take both Sobel x and y gradients
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel))
        sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel))
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= t_low) & (gradmag <= t_high)] = self.binary_value
        # Return the binary image
        return binary_output

    def apply_dir_sobel(self, image, kernel=3, t_low=0, t_high=np.pi/2):
        t_low_angle = t_low * np.pi/2
        t_high_angle = t_high * np.pi/2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= t_low_angle) & (absgraddir <= t_high_angle)] = self.binary_value
        # Return the binary image
        return binary_output