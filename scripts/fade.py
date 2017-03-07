from layer import Layer
import numpy as np

class Fade(Layer):

    grad_cover = None

    def __init__(self):
        self.name = "Fade"

    def process(self, image_input, image_original, model, output_path):
        if self.grad_cover is None:
            self.create_cover(image_input)

        image_input = image_input * self.grad_cover
        return image_input

    def create_cover(self, image_input):

        x_max = image_input.shape[1]
        x_mid = x_max / 2

        cover_start_pos = 0.15  # Position from the middle to start the fade
        cover_end_pos = 0.3  # % Position from middle to end the fade
        cover_start_value = 1.00
        cover_end_value = 0.05

        cover_fade_start_x = x_mid * cover_start_pos
        cover_fade_end_x = x_mid * cover_end_pos
        cover_fade_length = cover_fade_end_x - cover_fade_start_x

        self.grad_cover = np.zeros_like(image_input, dtype=np.float32)

        for x in range(0, x_max):
            x_from_mid = abs(x_mid - x)
            fade_factor = (x_from_mid - cover_fade_start_x) / cover_fade_length
            fade_factor = np.clip(fade_factor, 0, 1)
            self.grad_cover[:, x] = self.lerp(cover_start_value, cover_end_value, fade_factor)

    def lerp(self, a, b, f):
        return a + f * (b - a)