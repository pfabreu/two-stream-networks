import csv
import numpy as np
import cv2
import utils


def load_split(ids, labels, dim, n_channels, gen_type, of_len, first_epoch, encoding="rgb", soft_sigmoid=False):
    'Generates data containing batch_size samples'

    # TODO Warp
    root_dir = "/media/pedro/actv-ssd/flow_" + gen_type
    # Initialization, assuming its bidimensional (for now)
    X = np.empty([len(ids), dim[0], dim[1], n_channels])
    # Generate data
    sep = "@"
    ypose = np.empty(len(ids))
    yobject = []
    yhuman = []
    # Generate data
    for i, ID in enumerate(ids):
        # Get image from ID (since we are using opencv we get np array)
        split_id = ID.split(sep)
        vid_name = split_id[0]
        keyframe = split_id[1]
        vid_name = vid_name + "_" + keyframe
        # bbs = str(float(split_id[2])) + "_" + str(float(split_id[3])) + "_" + str(float(split_id[4])) + "_" + str(float(split_id[5]))
        frame = int(split_id[6])

        of_volume = np.zeros(shape=(dim[0], dim[1], n_channels))
        v = 0
        for fn in range(-of_len // 2, of_len // 2):
            of_frame = frame + fn
            if encoding == "grayscale":
                x_img_name = root_dir + "/x/" + vid_name + "/frame" + str('{:06}'.format(of_frame)) + ".jpg"
                x_img = cv2.imread(x_img_name, cv2.IMREAD_GRAYSCALE)
                if x_img is None:
                    if first_epoch:
                        with open("missing_files.txt", "a") as text_file:
                            text_file.write("%s" % x_img_name)
                    continue
                y_img_name = root_dir + "/y/" + vid_name + "/frame" + str('{:06}'.format(of_frame)) + ".jpg"
                y_img = cv2.imread(y_img_name, cv2.IMREAD_GRAYSCALE)
                if y_img is None:
                    continue

            elif encoding == "rgb":
                f_img_name = root_dir + "/" + vid_name + "/frame" + str('{:06}'.format(of_frame)) + ".jpg"

                f_img = cv2.imread(f_img_name)
                try:
                    # TODO this is an awful programming practice
                    # but it might be possible that some flow images don't have a last image (frame 36) due to opencv/ffmpeg imprecision
                    x_img = f_img[:, :, 0]
                    y_img = f_img[:, :, 1]
                except:
                    pass
                f_img = None
            # Put them in img_volume (x then y)
            of_volume[:, :, v] = x_img
            v += 1
            of_volume[:, :, v] = y_img
            v += 1
        X[i, ] = of_volume
        if gen_type != "test":
            ypose[i] = labels[ID]['pose']
            yobject.append(labels[ID]['human-object'])
            yhuman.append(labels[ID]['human-human'])

    # conversion to one hot is done after
    return X, ypose, yobject, yhuman


def get_AVA_set(classes, filename, soft_sigmoid=False):
    # Load all lines of filename
    id_list = []
    start_frame = 12
    end_frame = 32
    jump_frames = 5  # Keyframe will be 3
    sep = "@"
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
                ID = video + sep + kf_timestamp.lstrip("0") + \
                    sep + str(bb_top_x) + sep + str(bb_top_y) + sep + str(bb_bot_x) + sep + str(bb_bot_y) + sep + str(frame)
                id_list.append(ID)
    id_list = list(set(id_list))
    return id_list


def get_AVA_labels(classes, partition, set_type, filename, soft_sigmoid=False):
    sep = "@"  # Must not exist in any of the IDs

    labels = {}
    # Parse partition and create a correspondence to an integer in classes
    class_ids = classes['label_id']
    print("Generating labels: " + str(len(class_ids)))
    # Find entries in the csv that correspond
    start_frame = 12
    end_frame = 32
    jump_frames = 5  # Keyframe will be 3
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
