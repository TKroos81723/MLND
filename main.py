import numpy as np
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
import tensorflow as tf
from cv2 import cv2
import time
from lane_detection import Lanes, predict_lane
lanes = Lanes()

