import csv
import numpy as np
import cv2
import sys
import os
import glob


def get_AVA_classes(csv_filename):
    """
    Gets all classes from an AVA csv, format of classes is a dictionary with:
    classes['label_id'] has all class ids from 1-80
    classes['label_name'] has all class names (e.g bend/bow (at the waist))
    classes['label_type'] is either PERSON_MOVEMENT (1-14), OBJECT_MANIPULATION
    (15-63) or PERSON_INTERACTION (64-80)
    """
    classes = []
    with open(csv_filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        headers = next(csvReader)
        classes = {}
        for h in headers:
            classes[h] = []

        for row in csvReader:
            for h, v in zip(headers, row):
                classes[h].append(v)
    return classes


def load_split(ids, dim, n_channels, of_len, rgb_dir, flow_dir, context_dict):
    'Generates data containing batch_size samples'
    resize = False
    sep = "@"
    # Initialization, assuming its bidimensional (for now)
    X_rgb = np.empty([len(ids), dim[0], dim[1], 3])
    X_flow = np.empty([len(ids), dim[0], dim[1], 20])
    X_context = np.empty([len(ids), 720])

    # Generate data
    for i, ID in enumerate(ids):
        # Get image from ID (since we are using opencv we get np array)
        split_id = ID.split(sep)
        vid_name = split_id[0]
        keyframe = split_id[1]
        bb_top_x = float(split_id[2])
        bb_top_y = float(split_id[3])
        bb_bot_x = float(split_id[4])
        bb_bot_y = float(split_id[5])
        vid_name = vid_name + "_" + keyframe
        bbs = str(bb_top_x) + "_" + str(bb_top_y) + "_" + str(bb_bot_x) + "_" + str(bb_bot_y)
        rgb_frame = split_id[6]
        # Many names: 12 = 25 = 1, 17 = 35 = 2, 22 = 45 = 3, 27 = 55 = 4, 32 = 65 = 5
        of_frame = 12 + (int(rgb_frame) - 1) * 5  # conversion of rgb to of name format
        # Is this the correct format? Yes, the format has to use _
        img_name = rgb_dir + "/" + vid_name + "_" + bbs + "/frames" + rgb_frame + ".jpg"
        if not os.path.exists(img_name):
            print(img_name)
            print("[Error] File does not exist!")
            sys.exit(0)

        img = cv2.imread(img_name)
        if resize is True:
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        # Store sample
        X_rgb[i, ] = img

        context_key = vid_name + \
            "@" + str(bb_top_x) + "@" + str(bb_top_y) + "@" + str(bb_bot_x) + "@" + str(bb_bot_y)
        context_str = context_dict[context_key]
        X_context[i, ] = np.array(context_str.split(" "))

        of_volume = np.zeros(
            shape=(dim[0], dim[1], 20))
        v = 0
        for fn in range(-of_len // 2, of_len // 2):
            of_frame = of_frame + fn

            x_img_name = flow_dir + "/x/" + vid_name + "/frame" + str('{:06}'.format(of_frame)) + ".jpg"
            x_img = cv2.imread(x_img_name, cv2.IMREAD_GRAYSCALE)
            if x_img is None:
                continue
            y_img_name = flow_dir + "/y/" + vid_name + "/frame" + str('{:06}'.format(of_frame)) + ".jpg"
            y_img = cv2.imread(y_img_name, cv2.IMREAD_GRAYSCALE)
            if y_img is None:
                continue
            # Put them in img_volume (x then y)
            of_volume[:, :, v] = x_img
            v += 1
            of_volume[:, :, v] = y_img
            v += 1
        X_flow[i, ] = of_volume

    return X_rgb, X_flow, X_context


def get_AVA_set(classes, directory):
    sep = "@"
    id_list = []
    start_frame = 1
    end_frame = 5
    jump_frames = 1  # Keyframe will be 3
    # Load all lines of filename
    for d in glob.glob(directory + "/*"):
        if d != directory:
            print d
            row = d.rsplit("/", 1)[1]
            row = row.split("_")
            print row
            video = "_".join(row[:-5])
            print video
            kf_timestamp = row[-5]
            print kf_timestamp
            # action = row[6]
            bb_top_x = row[-4]
            bb_top_y = row[-3]
            bb_bot_x = row[-2]
            bb_bot_y = row[-1]
            # This is due to the behav of range
            for frame in range(start_frame, end_frame + jump_frames, jump_frames):
                # Append to the dictionary
                ID = video + sep + kf_timestamp.lstrip("0") + \
                    sep + str(bb_top_x) + sep + str(bb_top_y) + sep + str(bb_bot_x) + sep + str(bb_bot_y) + sep + str(frame)
                id_list.append(ID)

    return id_list
