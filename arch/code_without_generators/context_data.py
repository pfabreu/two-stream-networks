import csv
import numpy as np
import os.path
import sys
import utils


def load_split(ids, labels, dim, n_channels, gen_type, root_dir):
    'Generates data containing batch_size samples'
    sep = "@"
    X = np.empty([len(ids), dim])

    ypose = np.empty(len(ids))
    yobject = []
    yhuman = []
    Xfilename = root_dir + "context_files/XContext_" + gen_type + "_pastfuture.csv"
    if not os.path.exists(Xfilename):
        print("File does not exist")
        print(Xfilename)
        sys.exit(0)

    with open(Xfilename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            vid_name = row[0]
            kf = row[1]

            bbs = str("{:.3f}".format(float(row[2]))) + sep + str("{:.3f}".format(float(row[3]))) + sep + str("{:.3f}".format(float(row[4]))) + sep + str("{:.3f}".format(float(row[5])))
            xline_str = row[6]
            ID = vid_name + sep + kf.lstrip("0") + sep + bbs
            i = ids.index(ID)
            xline = np.array(xline_str.split(" "))

            X[i, ] = xline  # Convert xline to a numpy array
            ypose[i] = labels[ID]['pose']
            yobject.append(labels[ID]['human-object'])
            yhuman.append(labels[ID]['human-human'])

    # conversion to one hot is done after
    return X, ypose, yobject, yhuman


def get_AVA_set(classes, filename):
    sep = "@"
    id_list = []

    # Load all lines of filename
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            video = row[0]
            kf_timestamp = row[1]

            # action = row[6]
            # bb_top_x = row[2]
            # bb_top_y = row[3]
            # bb_bot_x = row[4]
            # bb_bot_y = row[5]
            bbs = str("{:.3f}".format(float(row[2]))) + sep + str("{:.3f}".format(float(row[3]))) + sep + \
                str("{:.3f}".format(float(row[4]))) + sep + str("{:.3f}".format(float(row[5])))

            ID = video + sep + kf_timestamp.lstrip("0") + sep + bbs
            id_list.append(ID)

    id_list = list(set(id_list))
    return id_list


def get_AVA_labels(classes, partition, set_type, filename):
    sep = "@"  # Must not exist in any of the IDs
    labels = {}
    # Parse partition and create a correspondence to an integer in classes
    class_ids = classes['label_id']
    print("Generating labels: " + str(len(class_ids)))
    # Find entries in the csv that correspond

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
#            bb_top_x = row[2]
#            bb_top_y = row[3]
#            bb_bot_x = row[4]
#            bb_bot_y = row[5]
#            bbs = str(bb_top_x) + sep + str(bb_top_y) + sep + str(bb_bot_x) + sep + str(bb_bot_y)
            bbs = str("{:.3f}".format(float(row[2]))) + sep + str("{:.3f}".format(float(row[3]))) + sep + str("{:.3f}".format(float(row[4]))) + sep + str("{:.3f}".format(float(row[5])))

            action = int(row[6])
            # Construct IDs
            label_ID = video + sep + kf.lstrip("0") + sep + bbs
            if action <= utils.POSE_CLASSES:
                labels[label_ID]['pose'] = action - 1
            elif action > utils.POSE_CLASSES and action <= utils.POSE_CLASSES + utils.OBJ_HUMAN_CLASSES:
                labels[label_ID]['human-object'].append(action - 1)
            else:
                labels[label_ID]['human-human'].append(action - 1)
    return labels
