# -*- coding: utf-8 -*-
# Go through every class folder, create an output class folder and run compute_flow for all those videos (do this 101 times)

import cv2
import os
import math
import glob

_VID_DIR = '/media/pedro/actv3/UCF101/videos/'
_OUT_DIR = "/media/pedro/actv3/UCF101/pose/"
OPENPOSE_DIR = "/home/pedro/openpose/"
MODELS_DIR = "../../arch/pose_stream/models"

# Load video
vid_count = 0
class_folders = glob.glob(_VID_DIR + "*")
continue_idx = 0  # set to 0 if you want ot process all videos

for c in class_folders: