import numpy as np
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
import tensorflow as tf
from cv2 import cv2
import time
from test import process, cal_curvature


# Class to average lanes with
class Lanes:
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


class model_config:
    def __init__(self):
        self.video_name = []
        self.width = []
        self.height = []
        self.average = []


paras = model_config()
paras.width = 852  # 1376
paras.height = 480  # 776
paras.video_name = "video/PM/vid_7.mp4"
paras.average = 20


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > paras.average:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (paras.height, paras.width, 3))

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    return lane_image, result


# Read lane realtime from webcam
model = load_model('full_CNN_model.h5')
cap = cv2.VideoCapture(paras.video_name)
start = time.time()
lanes = Lanes()
start_frame = time.time()
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        is_th, th_img = cv2.threshold(frame, 170, 255, cv2.THRESH_BINARY)
        cv2.imshow("THRESH", th_img)
        if i < paras.average:
            pre_out, output = road_lines(frame)
            i = i + 1
        else:
            pre_out, output = road_lines(frame)
            # output_data has radius and center_offset
            last_out, output_data = process(pre_out, output)
            stop_frame = time.time()
            FPS = 1 / (stop_frame - start_frame)
            print(FPS)
            start_frame = stop_frame
            cv2.waitKey(1)
    else:
        break
print(time.time() - start)
cap.release()
