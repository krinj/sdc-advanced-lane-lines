from layer import Layer
import numpy as np

class Overlay(Layer):

    def __init__(self):
        self.name = "Overlay"

    def process(self, image_input, image_original, model, output_path):

        alpha = 1
        beta = 0.8
        dst = image_original * alpha + image_input * beta;
        dst = np.clip(dst, 0, 255)

        return dst
