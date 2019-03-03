import cv2
import numpy as np
import np_opencv_module as npcv
from yolt_python import LaplacianBlending as fv
from matplotlib import pyplot as plt
from random import randint
import csv
import os
from PIL import Image
import gc
import pickle
import math

originalres = np.array([342, 256])
myres = np.array([224, 224])
resrefactor = np.array([342.0 / 224.0, 256.0 / 224.0])
root_dir = '/media/pedro/actv5/UCF101/jpegs_256/'
target_dir = '/media/pedro/actv-ssd/UCF101-24/fovea/'
# Load UCF Annotations

with open('../../../../data/UCF101-24/pyannot2.pkl', 'rb') as f:
    data = pickle.load(f)
#pickle.dump(data, open("ucfp2.pkl","wb"), protocol=2)
videoDirs = []
videos = []
labels = []
BBs = []  # [x y width height]
start_frames = []
end_frames = []
for key in data:
    dirinfo = key.split('/')
    for annot in data[key]['annotations']:
        videoDirs.append(dirinfo[0])
        videos.append(dirinfo[1])
        labels.append(annot['label'])
        start_frames.append(annot['sf'])
        end_frames.append(annot['ef'])
        BBs.append(annot['boxes'])

# Foveate 5 frames in each video per bounding box
for index in range(len(BBs)):
    if (index % 500 == 1):
        print "Now on frame #" + str(index) + " of " + str(len(BBs)) + ".\n"

    # Select which Frames to Foveate
    sf = start_frames[index]
    ef = end_frames[index]
    frame_interval = int(math.ceil((ef - sf) / 5.0))
    counter = 1
    for current_frame in range(sf, ef, frame_interval):
        current_BB = BBs[index][current_frame - sf]
        x_center = ((int(current_BB[0]) + int(current_BB[2])) / 2.0) / resrefactor[0]  # coordinates in resized image
        y_center = ((int(current_BB[1]) + int(current_BB[3])) / 2.0) / resrefactor[1]  # coordinates in resized image
        center = [int(x_center), int(y_center)]  # coordinates in resized image

        sigma_x = int(2 * current_BB[2] / resrefactor[0])
        sigma_y = int(2 * current_BB[3] / resrefactor[1])
        # Load current Frame
        current_frame = current_frame + 1
        frame_path = root_dir + videos[index] + '/frame' + str(current_frame + 1).zfill(6) + '.jpg'
        while not os.path.isfile(frame_path):
            current_frame = current_frame - 1
            frame_path = root_dir + videos[index] + '/frame' + str(current_frame).zfill(6) + '.jpg'
        if not os.path.isfile(frame_path):
            print "WTF IT DOES NOT EXIST!!!!"
        else:
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (myres[0], myres[1]))
            height, width, channels = img.shape
            img = plt.imread(frame_path)  # CHECK THAT IMG IS GETTING RESIZED
            img = cv2.resize(img, (myres[0], myres[1]))
            my_lap_obj = fv(width, height, 3, sigma_x, sigma_y)
            foveated_img = my_lap_obj.foveate(img, npcv.test_np_mat(np.array(center)))
            plt.cla()

            # Check that output Folder Exists, if not create it
            BB_counter = 0
            outdir = target_dir + videos[index] + '_BB_' + str(BB_counter)  # + '_' + str(current_BB[0]) + '_' + str(current_BB[1]) + '_' + str(current_BB[2]) + '_' + str(current_BB[3])
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            outimgpath = outdir + '/frame' + str(counter).zfill(1) + '.jpg'
            while os.path.isfile(outimgpath):
                outdir = target_dir + videos[index] + '_BB_' + str(BB_counter)
                outimgpath = outdir + '/frame' + str(counter).zfill(1) + '.jpg'
                BB_counter = BB_counter + 1
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            counter = counter + 1
            # Convert BGR to RGB
            #rgb_img = foveated_img[:, :, ::-1].copy()
            Image.fromarray(foveated_img).save(outimgpath)
        gc.collect()
