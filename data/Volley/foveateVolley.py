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
target_dir = '/home/jantunes/Documents/TensorFlowObjectDetection/models/research/object_detection/data/volleyball/fovea/'
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
                        x_center = ((int(current_BB[0]) + int(current_BB[2]))/2.0 )/resrefactor[0] #coordinates in resized image
                        y_center = ((int(current_BB[1]) + int(current_BB[3]))/2.0 )/resrefactor[1] #coordinates in resized image
                        center = [int(x_center),int(y_center)] #coordinates in resized image
                        
                        sigma_x = int(2*current_BB[2]/resrefactor[0]);
                        sigma_y = int(2*current_BB[3]/resrefactor[1]);
                        
                        frame_path = frame_dir + str(current_frame) + ".jpg"
                        
                        img = cv2.imread(frame_path);
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img,(myres[0],myres[1]))
                        height, width, channels = img.shape
                        img=plt.imread(frame_path)# CHECK THAT IMG IS GETTING RESIZED
                        img = cv2.resize(img,(myres[0],myres[1]))
                        my_lap_obj=fv(width,height,3,sigma_x,sigma_y)
                        foveated_img = my_lap_obj.foveate(img,npcv.test_np_mat(np.array(center)))
                        plt.cla()
                        
                        # Check that output Folder Exists, if not create it
                        BB_counter = 0;
                        outdir = target_dir + video_dir + '/' + str(frames[2]) + '_BB_' + str(current_BB_number)
                        if not os.path.isdir(outdir):
                            os.makedirs(outdir)
                        
                        outimgpath = outdir + '/frame' + str(counter).zfill(1) + '.jpg';
                        # Convert BGR to RGB
                        #rgb_img = foveated_img[:, :, ::-1].copy()
                        Image.fromarray(foveated_img).save(outimgpath);
                        counter = counter + 1

                current_BB_number = current_BB_number + 1
                gc.collect()

    gc.collect()
    video_counter = video_counter + 1



