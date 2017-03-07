from os import listdir
from cv2 import imread
from moviepy.editor import VideoFileClip

from pipeline import Pipeline
from model import Model
from warp import Warp
from calibrate import Calibrate
from colorthreshold import ColorThreshold
from lanefinder import LaneFinder
from fade import Fade
from overlay import Overlay
from measure import Measure

test_image_directory = "../test_images/"
cal_image_directory = "../camera_cal/"
output_image_directory = "../output_images/"

cal_shape = (9, 6)

def get_test_images():
    image_list = listdir(test_image_directory)
    return image_list

def create_pipeline(name="Unnamed Pipeline", output_step = 1):
    global pipeline
    pipeline = Pipeline(name=name, output_dir=output_image_directory, output_step=output_step)
    pipeline.add_layer(Calibrate(cal_image_directory, cal_shape))
    pipeline.add_layer(Warp())
    pipeline.add_layer(ColorThreshold())
    pipeline.add_layer(Fade())
    pipeline.add_layer(LaneFinder())
    pipeline.add_layer(Warp(inverse = True))
    pipeline.add_layer(Overlay())
    pipeline.add_layer(Measure())
    return pipeline

def process_images():
    global pipeline
    global model
    image_list = get_test_images()
    pipeline.should_save_images = True
    for i in image_list:
        image = imread(test_image_directory + i)
        pipeline.process(image, Model())

def process_video_image(video_image):
    global pipeline
    global  model
    pipeline.should_save_images = True
    pipeline.should_convert_to_BGR = True
    return pipeline.process(video_image, model)

def process_movie(input_path, output_path):
    clip = VideoFileClip(input_path)
    out_clip = clip.fl_image(process_video_image)  # NOTE: this function expects color images!!
    out_clip.write_videofile(output_path, audio=False)

if __name__ == "__main__":
    pipeline = create_pipeline("Perth Video Pipeline", 30)
    model = Model()
    #process_images()
    process_movie("../perth.mp4", "../output_videos/perth.mp4")
