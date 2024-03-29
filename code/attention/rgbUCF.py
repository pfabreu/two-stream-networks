import csv
import cv2
import os
import math
import numpy as np
import pickle
from PIL import Image, ImageFilter

originalres = np.array([342, 256])
myres = np.array([224, 224])
resrefactor = np.array([342.0 / 224.0, 256.0 / 224.0])
root_dir = '/media/pedro/actv5/UCF101/jpegs_256/'
target_dir = '/media/pedro/actv-ssd/UCF101-24/rgb/'
# Load UCF Annotations

with open('../../data/UCF101-24/pyannot2.pkl', 'rb') as f:
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


for index in range(len(BBs)):
    if (index % 1000 == 0):
        print "Now on frame #" + str(index) + " of " + str(len(BBs)) + ".\n"

    sf = start_frames[index]
    ef = end_frames[index]
    frame_interval = int(math.ceil((ef - sf) / 5.0))
    counter = 1
    for current_frame in range(sf, ef, frame_interval):
        current_BB = BBs[index][current_frame - sf]
        # Load current Frame
        current_frame = current_frame + 1
        frame_path = root_dir + videos[index] + '/frame' + str(current_frame + 1).zfill(6) + '.jpg'
        while not os.path.isfile(frame_path):
            current_frame = current_frame - 1
            frame_path = root_dir + videos[index] + '/frame' + str(current_frame).zfill(6) + '.jpg'
        if not os.path.isfile(frame_path):
            print "WTF IT DOES NOT EXIST!!!!"
        else:
            bx = (int(float(current_BB[0]) / (float(resrefactor[0]))), int(float(current_BB[1]) / float(resrefactor[1])), int((float(current_BB[0]) + float(current_BB[2])) / float(resrefactor[0])), int((float(current_BB[1]) * float(current_BB[3])) / float(resrefactor[1])))

            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (myres[0], myres[1]))

            # Check that output Folder Exists, if not create it
            outdir = target_dir + videos[index]
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            outimgpath = outdir + '/frame' + str(counter).zfill(1) + '.jpg'
            if not os.path.isfile(outimgpath):
                counter = counter + 1
                Image.fromarray(img).save(outimgpath)
