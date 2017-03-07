from cv2 import imwrite
import os
import cv2
class Pipeline(object):

    layers = []
    dir_path = ""
    current_image_id = 0
    output_step = 1
    should_save_images = True
    should_convert_to_BGR = False

    def __init__(self, name = "Pipeline", output_dir = "../output_images", output_step = 1):
        self.dir_path = output_dir + "/" + name + "/"
        self.output_step = output_step

    def add_layer(self, layer):
        layer.id = len(self.layers)
        self.layers.append(layer)

    def process(self, image, model):
        self.current_image_id += 1
        model.current_image_id = self.current_image_id
        print("Processing ", self.current_image_id)

        if self.should_convert_to_BGR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output_image = image

        for layer in self.layers:
            if self.should_save_images:
                self.create_output_dir(layer)
                layer.should_save = self.can_save_on_frame(self.current_image_id, self.output_step)
            else:
                layer.should_save = False

            output_image = layer.process(output_image, image, model, self.output_path_of_layer(layer))
            if layer.override_original:
                image = output_image
            self.save_image(output_image, layer)

        if self.should_convert_to_BGR:
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        return output_image

    def can_save_on_frame(self, frame_id, step):
        return frame_id % step == 0

    def create_output_dir(self, layer):
        if layer.should_save:
            final_dir = self.output_path_of_layer(layer)
            if not os.path.exists(final_dir):
                os.makedirs(final_dir)

    def save_image(self, image, layer):
        if layer.should_save and self.should_save_images:
            img_name = "image_" + str(self.current_image_id) + ".jpg"
            final_dir = self.output_path_of_layer(layer)
            draw_img = image.copy()
            draw_img = cv2.putText(draw_img, layer.name.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .85, [255, 255, 255], 1)
            imwrite(final_dir + img_name, draw_img);

    def output_path_of_layer(self, layer):
        return "{0}Layer{1}_{2}/".format(self.dir_path, layer.id, layer.name)