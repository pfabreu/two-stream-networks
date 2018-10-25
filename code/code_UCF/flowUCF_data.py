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


def make_chunks(original_list, size, chunk_size):
    seq = original_list[:size]
    splits = [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]
    return splits


def load_set(annot_path, setlist, index_path, datadir, dim, n_channels_of, of_length):
    'Load X and Y for a set'

    # Load annotations and set list
    with open(annot_path, 'rb') as f:
        annots = pickle.load(f)
    with open(index_path, 'rb') as f:
        frame_indexes = pickle.load(f)
    keys = []
    for st in setlist:
        keys.append(st.split(' ')[0][:-4])
    X = np.empty([3000, dim[0], dim[1], n_channels_of])  # overshooting size. will cut to right size at the end of this function.
    y = np.empty(5000)  # overshooting size. will cut to right size at the end of this function.
    dataIndex = 0
    for key in keys:
        if key in annots:
            dirinfo = key.split('/')
            for BB in range(len(annots[key]['annotations'])):
                index_key = dirinfo[1] + "_BB_" + str(BB)
                current_indexes = frame_indexes[index_key]
                udir = datadir + "/u/" + dirinfo[1] + "/"
                vdir = datadir + "/v/" + dirinfo[1] + "/"
                label = annots[key]['annotations'][BB]['label']
                for img_number in current_indexes:
                    # Load the 20 images corresponding to the current rgb frame
                    v = 0
                    of_volume = np.zeros([dim[0], dim[1], n_channels_of])
                    for fn in range(-of_length // 2, of_length // 2):
                        current_frame = img_number + fn
                        uimg_path = udir + "frame" + str(current_frame).zfill(6) + ".jpg"
                        vimg_path = vdir + "frame" + str(current_frame).zfill(6) + ".jpg"
                        if not os.path.exists(uimg_path):
                            u_img = np.zeros([dim[0], dim[1]])
                        else:
                            u_img = cv2.resize(cv2.imread(uimg_path, cv2.IMREAD_GRAYSCALE), (dim[0], dim[1]))
                        if not os.path.exists(vimg_path):
                            v_img = np.zeros([dim[0], dim[1]])
                        else:
                            v_img = cv2.resize(cv2.imread(vimg_path, cv2.IMREAD_GRAYSCALE), (dim[0], dim[1]))
                        of_volume[:, :, v] = u_img
                        v += 1
                        of_volume[:, :, v] = v_img
                        v += 1
                    X[dataIndex, ] = of_volume
                    y[dataIndex] = label
                    dataIndex += 1
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
