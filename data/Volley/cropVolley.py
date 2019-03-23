import cv2
import numpy as np
import np_opencv_module as npcv
from yolt_python import LaplacianBlending as fv
from matplotlib import pyplot as plt
from random import randint
import csv
import os
from PIL import Image
from sendemail import *
import gc
import pickle
import math

originalres = np.array([1280,720]);
myres = np.array([224,224]);
resrefactor = np.array([1280.0/224.0,720.0/224.0])
root_dir = '/home/jantunes/Documents/TensorFlowObjectDetection/models/research/object_detection/data/volleyball/videos/'
target_dir = '/home/jantunes/Documents/TensorFlowObjectDetection/models/research/object_detection/data/volleyball/crop/'
# Load UCF Annotations

video_dirs = os.listdir(root_dir)
video_counter = 1;
for video_dir in video_dirs:
    print "Now on Video " + str(video_counter) + " of " + str(len(video_dirs)) + ".\n"
    if os.path.isdir(root_dir + video_dir):
        current_video_dir = root_dir + video_dir + "/"
        text_file = open(current_video_dir + "annotations.txt", "r")
        video_labels_list = text_file.readlines()
        for video_labels in video_labels_list:
            video_labels = video_labels.split(" ")
            frame_number = int(video_labels[0][:-4])
            frame_dir = current_video_dir + video_labels[0][:-4] + "/";
            video_labels = video_labels[2:]
            blocksize= 5
            current_BB_number = 0
            if( not os.path.exists(target_dir + video_dir)):
                os.makedirs(target_dir + video_dir)
            for i in xrange(0, len(video_labels), blocksize):
                if((i+blocksize) < len(video_labels)):
                    chunk = video_labels[i:i+blocksize] # BB is [X Y W H]
                    current_BB = [int(chunk[0]), int(chunk[1]), int(chunk[2]), int(chunk[3]) ] # BB is [X Y W H]
                    frames = [ frame_number-20, frame_number-10, frame_number, frame_number+10, frame_number+20]
                    counter = 0
                    for current_frame in frames:
                        
                        bx = (int(float(current_BB[0])/(float(resrefactor[0]))), int(float(current_BB[1])/float(resrefactor[1])), int((float(current_BB[0]) + float(current_BB[2]))/float(resrefactor[0])), int((float(current_BB[1]) + float(current_BB[3]))/float(resrefactor[1])))
                        #bx = (int(float(current_BB[0])), int(float(current_BB[1])), int(float(current_BB[0]) + float(current_BB[2])), int(float(current_BB[1]) + float(current_BB[3])))

                        frame_path = frame_dir + str(current_frame) + ".jpg"
                        
                        img = cv2.imread(frame_path);
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img,(myres[0],myres[1]))
                        img = Image.fromarray(img)
                        box = img.crop(bx)
                        img = Image.new('RGB',(224,224))
                        # Blur the whole image
                        #img = img.filter(ImageFilter.GaussianBlur(10))
                        img.paste(box, bx)  # Paste them back
                        img = img.resize((224, 224), Image.NEAREST)
                        
                        # Check that output Folder Exists, if not create it
                        BB_counter = 0;
                        outdir = target_dir + video_dir + '/' + str(frames[2]) + '_BB_' + str(current_BB_number)
                        if not os.path.isdir(outdir):
                            os.makedirs(outdir)
                        
                        outimgpath = outdir + '/frame' + str(counter).zfill(1) + '.jpg';
                        # Convert BGR to RGB
                        #rgb_img = foveated_img[:, :, ::-1].copy()
                        img.save(outimgpath)
                        counter = counter + 1

                current_BB_number = current_BB_number + 1
                gc.collect()

    gc.collect()
    video_counter = video_counter + 1



