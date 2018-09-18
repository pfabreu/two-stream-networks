"""
Class for managing our data.
"""
import csv
import numpy as np
import cv2
import os.path
import sys
import utils
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import random


def load_split(ids, labels, dim, n_channels, gen_type, filter_type, soft_sigmoid=False, train=True):
    'Generates data containing batch_size samples'
    resize = False
    sep = "@"
    X = np.zeros([len(ids), dim[0], dim[1], n_channels])
    rgb_dir = "/media/pedro/actv-ssd/"
    # rgb_dir = ""
    ypose = np.empty(len(ids))
    yobject = []
    yhuman = []
    # Generate data
    for i, ID in enumerate(ids):
        # Get image from ID (since we are using opencv we get np array)
        split_id = ID.split(sep)
        vid_name = split_id[0]
        trueID = ID
        if vid_name[0] != '#':

            keyframe = split_id[1]
            vid_name = vid_name + "_" + keyframe
            bbs = str(float(split_id[2])) + "_" + str(float(split_id[3])) + "_" + str(float(split_id[4])) + "_" + str(float(split_id[5]))
            rgb_frame = split_id[6]

            # Is this the correct format? Yes, the format has to use _
            img_name = rgb_dir + filter_type + "_" + gen_type + "/" + vid_name + "_" + bbs + "/frames" + rgb_frame + ".jpg"

            if not os.path.exists(img_name):
                img = np.zeros((224, 224, 3))
                print(img_name)
                print("[Error] File does not exist... Using a black image instead")
                # sys.exit(0)
            else:
                img = cv2.imread(img_name)
                if resize is True:
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

        else:
            trueID = ID[1:]
            vid_name = vid_name[1:]
            # print(ID)
            # print(vid_name)
            # Store sample
            keyframe = split_id[1]
            vid_name = vid_name + "_" + keyframe
            bbs = str(float(split_id[2])) + "_" + str(float(split_id[3])) + "_" + str(float(split_id[4])) + "_" + str(float(split_id[5]))
            rgb_frame = split_id[6]

            # Is this the correct format? Yes, the format has to use _
            img_name = rgb_dir + filter_type + "_" + gen_type + "/" + vid_name + "_" + bbs + "/frames" + rgb_frame + ".jpg"
            img = cv2.imread(img_name)
            # print(img_name)
            # print(img.shape)
            if random.random() < 0.5:
                img = np.fliplr(img)
            crop_rand_val = random.randrange(0, 5, 1) / 10.0
            #scale_rand_val = random.randrange(7, 10, 1) / 10.0
            # print(crop_rand_val)
            # print(scale_rand_val)
            seq = iaa.Sequential([  # horizontal flips
                #iaa.Scale((scale_rand_val, 1.0)),
                iaa.CropAndPad(
                    percent=(0, crop_rand_val),
                    pad_mode=["edge"]
                )  # random crops
            ], random_order=True)  # apply augmenters in random order

            img = seq.augment_image(img)

        X[i, ] = img
        if train is True:
            ypose[i] = labels[trueID]['pose']
            yobject.append(labels[trueID]['human-object'])
            yhuman.append(labels[trueID]['human-human'])

            # conversion to one hot is done after
    return X, ypose, yobject, yhuman


def get_AVA_set(classes, filename, soft_sigmoid=False):
    sep = "@"
    id_list = []
    start_frame = 1
    end_frame = 5
    jump_frames = 1  # Keyframe will be 3

    # Load all lines of filename
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            video = row[0]
            kf_timestamp = row[1]
            # action = row[6]
            bb_top_x = row[2]
            bb_top_y = row[3]
            bb_bot_x = row[4]
            bb_bot_y = row[5]
            # This is due to the behav of range
            for frame in range(start_frame, end_frame + jump_frames, jump_frames):
                # Append to the dictionary
                ID = video + sep + kf_timestamp.lstrip("0") + sep + str(bb_top_x) + sep + str(bb_top_y) + sep + str(bb_bot_x) + sep + str(bb_bot_y) + sep + str(frame)
                id_list.append(ID)
    id_list = list(set(id_list))  # Make sure we only got unique id's
    return id_list


def get_AVA_labels(classes, partition, set_type, filename, soft_sigmoid=False):
    sep = "@"  # Must not exist in any of the IDs

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
