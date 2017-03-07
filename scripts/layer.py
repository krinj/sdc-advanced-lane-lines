class Layer(object):

    name = "Base"
    id = 0
    should_save = True
    override_original = False # If true, this output will override "original image" in the pipeline.

    def __init__(self):
        pass

    def process(self, image_input, image_original, model, output_path):
        return image_input