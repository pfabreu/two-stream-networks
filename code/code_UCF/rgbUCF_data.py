"""
Class for managing our data.
"""
import csv
import numpy as np
import cv2
import os.path
import sys
#import utils
import pickle


def load_set(annot_path, setlist_path, datadir, dim, n_channels):
    'Load X and Y for a set'

    # Load annotations and set list
    with open(annot_path, 'rb') as f:
        annots = pickle.load(f)
    with open(setlist_path, 'rb') as f:
        setlist = pickle.load(f)
    keys = []
    for st in setlist:
        keys.append(st.split(' ')[0][:-4])
    X = np.zeros([18000, dim[0], dim[1], n_channels])  # overshooting size. will cut to right size at the end of this function.
    y = np.zeros(18000)  # overshooting size. will cut to right size at the end of this function.
    dataIndex = 0
    for key in keys:
        if key in annots:
            dirinfo = key.split('/')
            for BB in range(len(annots[key]['annotations'])):
                if datadir == "../../../UCF_rgb/":
                    frame_path = dirinfo[1] + "/"
                else:
                    frame_path = dirinfo[1] + "_BB_" + str(BB) + "/"
                label = annots[key]['annotations'][BB]['label']
                for img_number in range(1, 6):
                    # Load the 5 images!
                    img_path = datadir + frame_path + "frame" + str(img_number) + ".jpg"
                    if not os.path.exists(img_path):
                        print(img_path)
                        print("[Error] File does not exist!")

                    else:
                        img = cv2.imread(img_path)
                        X[dataIndex, ] = img
                        y[dataIndex] = label
                        dataIndex = dataIndex + 1

        else:
            #print(key + " is not in our annotations. Skipping!\n")
            pass

    # Clip X and y to correct lengths
    X = X[:dataIndex, :, :, :]
    y = y[:dataIndex]
    return X, y


def get_AVA_labels(classes, partition, set_type, filename, soft_sigmoid=False):
    sep = "@"  # Must not exist in any of the IDs
    if soft_sigmoid is False:
        labels = {}
        # Parse partition and create a correspondence to an integer in classes
        class_ids = classes['label_id']
        print("Generating labels: " + str(len(class_ids)))
        # First process the training
        for entry in partition[set_type]:
            labels[entry] = int(entry.split('_')[-2]) - 1
    else:
        labels = {}
        # Parse partition and create a correspondence to an integer in classes
        class_ids = classes['label_id']
        print("Generating labels: " + str(len(class_ids)))
        # Find entries in the csv that correspond
        start_frame = 1
        end_frame = 5
        jump_frames = 1  # Keyframe will be 3
        for entry in partition[set_type]:
            labels[entry] = {}
            labels[entry]['pose'] = -1  # It might as well be a single entry here and not a list
            labels[entry]['human-object'] = []
            labels[entry]['human-human'] = []
        with open(filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                # Read rows
                video = row[0]
                kf = row[1]
                bb_top_x = row[2]
                bb_top_y = row[3]
                bb_bot_x = row[4]
                bb_bot_y = row[5]
                bbs = str(bb_top_x) + sep + str(bb_top_y) + sep + str(bb_bot_x) + sep + str(bb_bot_y)
                action = int(row[6])
                # Construct IDs
                for frame in range(start_frame, end_frame + jump_frames, jump_frames):
                    label_ID = video + sep + kf.lstrip("0") + sep + bbs + sep + str(frame)
                    if action <= utils.POSE_CLASSES:
                        labels[label_ID]['pose'] = action - 1
                    elif action > utils.POSE_CLASSES and action <= utils.POSE_CLASSES + utils.OBJ_HUMAN_CLASSES:
                        labels[label_ID]['human-object'].append(action - 1)
                    else:
                        labels[label_ID]['human-human'].append(action - 1)
    return labels
