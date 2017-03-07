from layer import Layer
import matplotlib.pyplot as plt
import numpy as np
import cv2

class LaneFinder(Layer):

    nonzero = []
    nonzeroy = []
    nonzerox = []

    right_lane_inds = []
    left_lane_inds = []

    def __init__(self):
        self.name = "LaneFinder"

    def process(self, image_input, image_original, model, output_path):

        # Create Plot Y Linspace
        if model.ploty is None:
            model.ploty = np.linspace(0, image_input.shape[0] - 1, image_input.shape[0])

        # Work out the histogram
        histogram = np.sum(image_input[image_input.shape[0] / 2:, :] / 255, axis=0)
        model.add_histogram(histogram)
        self.plot_histogram(model.histogram, output_path, model.current_image_id)

        # Create an output image to draw on and  visualize the result
        out_temp = np.dstack((image_input, image_input, image_input)) / 255
        out_img = np.zeros_like(out_temp)

        # Identify the x and y positions of all nonzero pixels in the image
        self.nonzero = image_input.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])
        self.left_lane_inds = []
        self.right_lane_inds = []

        window_img = None

        # Find the peaks
        start_left, start_right = self.find_peaks(image_input, model, histogram, output_path, model.current_image_id)

        if model.left_lane.confidence + model.right_lane.confidence > 1.35:
            window_img = self.roi_search(model, out_img)
        else:
            self.sliding_window_search(image_input, start_left, start_right, out_img)

        self.find_poly_fit(model)
        model.normalize_lanes()
        self.display_lane_pixels(out_img)
        self.display_poly_fit(model, out_img, output_path)
        out_img = out_img * 255

        highlight_img = self.draw_highlight(model, out_img)
        dst = out_img + highlight_img;

        if window_img is not None:
            dst = dst + window_img * 0.5;

        out_temp = out_temp * 150 + out_img
        out_temp = np.clip(out_temp, 0, 255)
        model.overlay_image = out_temp

        return dst

    def find_peaks(self, image_input, model, histogram, output_path, id):
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    def sliding_window_search(self, image_input, leftx_base, rightx_base, out_img):

        nwindows = 20

        window_height = np.int(image_input.shape[0] / nwindows)
        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = 40
        minpix = 25

        self.left_lane_inds = []
        self.right_lane_inds = []

        for window in range(nwindows):
            win_y_low = image_input.shape[0] - (window + 1) * window_height
            win_y_high = image_input.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 1, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 1, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low) & (
                self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) & (
                self.nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)
        pass

    def roi_search(self, model, out_img):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        margin = 35
        self.left_lane_inds = (
        (self.nonzerox > (model.left_lane.fit[0] * (self.nonzeroy ** 2) + model.left_lane.fit[1] * self.nonzeroy + model.left_lane.fit[2] - margin)) & (
        self.nonzerox < (model.left_lane.fit[0] * (self.nonzeroy ** 2) + model.left_lane.fit[1] * self.nonzeroy + model.left_lane.fit[2] + margin)))

        self.right_lane_inds = (
        (self.nonzerox > (model.right_lane.fit[0] * (self.nonzeroy ** 2) + model.right_lane.fit[1] * self.nonzeroy + model.right_lane.fit[2] - margin)) & (
            self.nonzerox < (model.right_lane.fit[0] * (self.nonzeroy ** 2) + model.right_lane.fit[1] * self.nonzeroy + model.right_lane.fit[2] + margin)))

        window_img = np.zeros_like(out_img)

        draw_margin = 5
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([model.left_lane.fitx - draw_margin, model.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([model.left_lane.fitx + draw_margin, model.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([model.right_lane.fitx - draw_margin, model.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([model.right_lane.fitx + draw_margin, model.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 100, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (255, 100, 0))
        return window_img

    def find_poly_fit(self, model):

        # Extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        # Fit a second order polynomial to each
        if len(lefty) == 0:
            model.left_lane.decay_confidence()
        else:
            p, res, _, _, _ = np.polyfit(lefty, leftx, 2, full=True)
            model.add_left_fit(p, res, len(leftx))

        if len(righty) == 0:
            model.right_lane.decay_confidence()
        else:
            p, res, _, _, _ = np.polyfit(righty, rightx, 2, full=True)
            model.add_right_fit(p, res, len(rightx))

        left_fit = model.left_lane.fit
        right_fit = model.right_lane.fit
        return left_fit, right_fit

    def display_poly_fit(self, model, out_img, output_path):
        # Display Results
        ploty = model.ploty
        left_fitx = model.left_lane.fitx
        right_fitx = model.right_lane.fitx

        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.savefig(output_path + "out{0}.png".format(model.current_image_id))
        plt.clf()

    def display_lane_pixels(self, out_img):
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0.5, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [255, 0.5, 0]

    def draw_highlight(self, model, out_img):
        # Create an image to draw the lines on
        highlight_img = np.zeros_like(out_img)

        # Display Results
        ploty = model.ploty
        left_fitx = model.left_lane.fitx
        right_fitx = model.right_lane.fitx

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(highlight_img, np.int_([pts]), (100, 25, 0))
        return highlight_img

    def plot_histogram(self, histogram, output_path, id = 0):
        if self.should_save:
            plt.plot(histogram)
            plt.savefig(output_path + "plot{0}.png".format(id))
            plt.clf()
