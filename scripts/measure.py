from layer import Layer
import numpy as np
import cv2

class Measure(Layer):

    def __init__(self):
        self.name = "Measure"

    def process(self, image_input, image_original, model, output_path):

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / model.lane_width_in_pixels  # meters per pixel in x dimension
        y_eval = np.max(model.ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(model.ploty * ym_per_pix, model.left_lane.fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(model.ploty * ym_per_pix, model.right_lane.fitx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # Find the lane positions and try to determine offset.
        histogram = model.histogram
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        lane_width = rightx_base - leftx_base
        lane_mid = self.lerp(leftx_base, rightx_base, 0.5)
        distance_from_center = lane_mid - midpoint
        meter_distance_from_center = xm_per_pix * distance_from_center * (1/model.zoom_factor)

        model.curve_radius = self.lerp(left_curverad, right_curverad, 0.5)
        model.distance_from_center = meter_distance_from_center

        h = image_input.shape[0]
        w = image_input.shape[1]
        image_input = self.draw_text(image_input, "Curve Radius: {0}m".format(self.shorten(model.curve_radius)), (50, 60), (380, 20))

        if model.distance_from_center < 0:
            image_input = self.draw_text(image_input,
                                         "{0}m Right of Center".format(self.shorten(-model.distance_from_center)),
                                         (50, 120), (380, 20))
        else:
            image_input = self.draw_text(image_input,
                                         "{0}m Left of Center".format(self.shorten(model.distance_from_center)),
                                         (50, 120), (380, 20))

        image_input = self.draw_text(image_input, "L. Conf: {0}".format(self.shorten(model.left_lane.confidence)), (0 + 20, h - 50), (180, 20), self.get_color(model.left_lane.confidence))
        image_input = self.draw_text(image_input, "R. Conf: {0}".format(self.shorten(model.right_lane.confidence)), (w-180 - 20, h - 50), (180, 20), self.get_color(model.right_lane.confidence))

        if model.overlay_image is not None:
            self.map_image(image_input, model.overlay_image)

        return image_input

    def lerp(self, a, b, f):
        return a + f * (b - a)

    def shorten(self, val):
        return int((val * 100) + 0.5) / 100.0

    def get_color(self, val):
        if val > 0.75:
            return [0, 255, 0]
        if val > 0.5:
            return [0, 255, 255]
        return [0, 0, 255]

    def draw_text(self, image, text, position, size, color = [255, 255, 255]):
        r_height = size[1]
        r_width = size[0]
        r_padding = 12
        r_pos1 = (position[0] - r_padding, position[1] + r_padding)

        r_pos2 = (position[0] + r_width + r_padding, position[1] - r_height - r_padding)
        image = cv2.rectangle(image, r_pos1, r_pos2, (0, 0, 0), thickness=-1)
        image = cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, .85, color, 2)
        return image

    def map_image(self, out_img, ovr_img):
        ovr_img = cv2.resize(ovr_img, (0, 0), fx=0.25, fy=0.25)
        x_offset = out_img.shape[1] - ovr_img.shape[1] - 20
        y_offset = 20
        out_img[y_offset:y_offset + ovr_img.shape[0], x_offset:x_offset + ovr_img.shape[1]] = ovr_img