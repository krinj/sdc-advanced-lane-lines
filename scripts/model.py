import numpy as np

class Model(object):

    histogram = None
    current_image_id = 0

    right_lane = None
    left_lane = None

    ploty = None

    zoom_factor = 0.25 # How far to zoom out when unwrapping
    lane_width_in_pixels = 700

    curve_radius = 0
    distance_from_center = 0

    overlay_image = None

    def __init__(self):
        self.right_lane = Lane("Right")
        self.left_lane = Lane("Left")

    def add_histogram(self, input_histogram):
        if self.histogram is None:
            self.histogram = input_histogram
        else:
            self.decay_histogram()
            self.histogram += input_histogram

    def decay_histogram(self):
        self.histogram = self.histogram / 2

    def add_left_fit(self, value, error, num_pix):
        return self.left_lane.add_fit(value, error, num_pix, self.ploty)

    def add_right_fit(self, value, error, num_pix):
        return self.right_lane.add_fit(value, error, num_pix, self.ploty)

    def normalize_lanes(self):
        if ((self.left_lane.confidence < 0.5) & (self.right_lane.confidence > 0.9)):
            self.left_lane.normalize_to(self.right_lane, 0.1, self.lane_width_in_pixels, self.zoom_factor, -1)

        if ((self.right_lane.confidence < 0.5) & (self.left_lane.confidence > 0.9)):
             self.right_lane.normalize_to(self.left_lane, 0.1, self.lane_width_in_pixels, self.zoom_factor, 1)

class Lane(object):

    fit = None
    fitx = None
    confidence = 0
    filter_rate = 0.5 # How much of the new value to add
    name = ""

    def __init__(self, name="Lane"):
        self.name = name
        pass

    def add_fit(self, value, error, num_pix, ploty):
        if self.fit is None:
            self.fit = value
            self.confidence = 1
        else:
            if self.detect_outlier(value, error, num_pix):
                # Outlier value is detected, ignore this input.
                self.decay_confidence()
            else:
                if len(error) > 0:
                    error = error[0] / num_pix
                    print("{0} Error: {1}, Num Px: {2}".format(self.name, error, num_pix))

                    if (error < 10) & (num_pix > 250):
                        self.confidence += 0.05

                    if (error < 15) & (num_pix > 500):
                        self.confidence += 0.05

                    if (error < 50) & (num_pix > 1000):
                        self.confidence += 0.1

                if self.confidence > 1:
                    self.confidence = 1

                self.fit = self.fit * (1 - self.filter_rate) + value * self.filter_rate

        self.fitx = self.fit[0] * ploty ** 2 + self.fit[1] * ploty + self.fit[2]
        return self.fit

    def decay_confidence(self):
        self.confidence = self.confidence * 0.95

    def detect_outlier(self, value, error, num_pix):

        if self.confidence < 0.25:
            return False

        difference = self.fit - value
        difference = difference ** 2
        d_sum = np.sum(difference)

        if d_sum > 1500:
            print("OUTLIER: ", d_sum)
            return True

        if len(error) > 0:
            error = error[0] / num_pix
            if (error > 50) & (num_pix < 500):
                print("OUTLIER: ", error, num_pix)
                return True

        return False

    def normalize_to(self, lane, factor, lane_width_in_pixels, zoom_factor, dir):
        self.fit[0] = lane.fit[0] * factor + self.fit[0] * (1 - factor)
        self.fit[1] = lane.fit[1] * factor + self.fit[1] * (1 - factor)
        self.fit[2] = ((lane.fit[2] + (dir * lane_width_in_pixels * zoom_factor)) * factor) + self.fit[2] * (1 - factor)

