# Go through all videos in AHA and for each get the fisher vectors
from __future__ import absolute_import, division, print_function
from datetime import datetime
from cv2 import DualTVL1OpticalFlow_create as DualTVL1
from tensorflow.python.platform import app, flags
import os
import sys
import cv2
import threading
import glob
import tensorflow as tf
import numpy as np
import subprocess


OUTPUT_DIR = '/media/pedro/actv4/aha_idt'
DATA_DIR = '/media/pedro/actv4/AHA_FMH/'
IDT_ROOT = '~/ei3d/architecture/iDT/release/'
_CLASS_NAMES = 'class_names.txt'

#vid_name = "sample"
#f = open(vid_name + '_features.gz', 'w')
#proc1 = subprocess.Popen(["./DenseTrackStab", vid_name + ".avi"], stdout=subprocess.PIPE)
#proc2 = subprocess.Popen("gzip", stdin=proc1.stdout, stdout=f)
# proc2.communicate()
# f.close()


def _process_dataset():
    videos = []

    filenames = [filename for class_fold in tf.gfile.Glob(os.path.join(
        DATA_DIR, '*')) for filename in tf.gfile.Glob(os.path.join(class_fold, '*'))]
    for f in filenames:
        video_name = f.split("/")[-1]

        if ".avi" in video_name:
            print(video_name)
            path_splits = f.split("/")[:-1]
            # Remove the .mat.avi and add features.gz
            output_name = video_name[:-8]
            output_name = "/" + path_splits[1] + "/" + path_splits[2] + "/" + path_splits[
                3] + "/" + "aha_idt" + "/" + path_splits[5] + "/" + output_name + "_features.gz"
            print(output_name)
            print(len(output_name))
            # output_name is output features
            feat_out = open(output_name, 'w')
            # f is input video
            proc1 = subprocess.Popen(
                [IDT_ROOT + "./DenseTrackStab", f], stdout=subprocess.PIPE)
            proc2 = subprocess.Popen(
                "gzip", stdin=proc1.stdout, stdout=feat_out)
            proc2.communicate()
            feat_out.close()
            videos.append(f)


def main():
    # Create directories for the classes
    if not os.path.exists(OUTPUT_DIR):
        f = open("class_names.txt")
        for line in f.read().splitlines():
            os.makedirs(OUTPUT_DIR + "/" + line + "/")
        f.close()

    # Process dataset
    _process_dataset()

if __name__ == '__main__':
    main()
