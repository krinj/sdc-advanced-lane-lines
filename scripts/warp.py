from layer import Layer
import numpy as np
import cv2

class Warp(Layer):

    transform_matrix = None
    matrix_calculated = False
    inverse = False

    def __init__(self, inverse=False):
        self.name = "Warp"
        self.inverse = inverse

    def process(self, image_input, image_original, model, output_path):

        if not self.matrix_calculated:
            self.calculate_matrix(image_input, model)

        img_size = (image_input.shape[1], image_input.shape[0])
        output_image = cv2.warpPerspective(image_input, self.transform_matrix, img_size, flags=cv2.INTER_LINEAR)
        return output_image

    def calculate_matrix(self, image_input, model):

        img_y = image_input.shape[0]
        img_x = image_input.shape[1]

        src = np.zeros((4, 2), dtype=np.float32)
        dst = np.zeros((4, 2), dtype=np.float32)

        x_top_factor = 0.035
        x_bot_factor = 0.300
        y_top_factor = 0.625
        y_bot_factor = 0.945
        x_adder = model.zoom_factor

        y_top = img_y * y_top_factor
        y_bot = img_y * y_bot_factor

        x_mid = img_x * 0.5
        x_top_left = x_mid - (x_top_factor * img_x)
        x_top_right = x_mid + (x_top_factor * img_x)
        x_bot_left = x_mid - (x_bot_factor * img_x)
        x_bot_right = x_mid + (x_bot_factor * img_x)
        x_bot_left_full = x_mid - (x_bot_factor * x_adder * img_x)
        x_bot_right_full = x_mid + (x_bot_factor * x_adder * img_x)

        src[0, 0], src[0, 1] = x_top_left, y_top
        src[1, 0], src[1, 1] = x_top_right, y_top
        src[2, 0], src[2, 1] = x_bot_right, y_bot
        src[3, 0], src[3, 1] = x_bot_left, y_bot

        dst[0, 0], dst[0, 1] = x_bot_left_full, 0
        dst[1, 0], dst[1, 1] = x_bot_right_full, 0
        dst[2, 0], dst[2, 1] = x_bot_right_full, img_y
        dst[3, 0], dst[3, 1] = x_bot_left_full, img_y

        if self.inverse:
            self.transform_matrix = cv2.getPerspectiveTransform(dst, src)
        else:
            self.transform_matrix = cv2.getPerspectiveTransform(src, dst)

        self.matrix_calculated = True

