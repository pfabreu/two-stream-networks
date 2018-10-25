import os
CPU = True
if CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # This must be imported before keras
import tensorflow as tf
import utils
import csv
import voting
import time
from keras import backend as K
import numpy as np
import pickle
from random import randint


def get_AVA_set(classes, filename, soft_sigmoid=False):
    sep = "@"
    id_list = []

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

            ID = video + sep + kf_timestamp.lstrip("0") + sep + str(bb_top_x) + sep + str(bb_top_y) + sep + str(bb_bot_x) + sep + str(bb_bot_y)
            id_list.append(ID)
    id_list = list(set(id_list))  # Make sure we only got unique id's
    return id_list


def main():
    K.clear_session()

    root_dir = '../../data/AVA/files/'

    # Load list of action classes and separate them (from utils_stream)
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')
    split = 'test'
    # Get validation set from directory
    partition = {}
    partition['test'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_" + split.title() + "_Custom_Corrected.csv", soft_sigmoid=True)

    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    result_csv = "test_outputs/random/output_test_random_" + time_str + ".csv"

    print("Test set size: " + str(len(partition['test'])))

    # When you're done getting all the votes, write output csv
    with open(result_csv, "a") as output_file:
        for row in partition['test']:
            row = row.split("@")
            video_name = row[0]
            timestamp = row[1]
            # action = row[6]
            bb_topx = row[2]
            bb_topy = row[3]
            bb_botx = row[4]
            bb_boty = row[5]
            # Generate a random pose guess
            rand_pose = randint(1, 10)
            line = video_name + "," + timestamp + "," + bb_topx + "," + bb_topy + "," + bb_botx + "," + bb_boty + "," + str(rand_pose)
            output_file.write("%s\n" % line)
            # Generate between 0 to 3 random human-object guesses
            rand_ho = randint(0, 3)
            for r in range(rand_ho):
                rand_humanobject = randint(11, 22)
                line = video_name + "," + timestamp + "," + bb_topx + "," + bb_topy + "," + bb_botx + "," + bb_boty + "," + str(rand_humanobject)
                output_file.write("%s\n" % line)
            # Generate between 0 to 3 random human-human guesses
            rand_hh = randint(0, 3)
            for r in range(rand_hh):
                rand_humanhuman = randint(23, 30)
                line = video_name + "," + timestamp + "," + bb_topx + "," + bb_topy + "," + bb_botx + "," + bb_boty + "," + str(rand_humanhuman)
                output_file.write("%s\n" % line)

if __name__ == '__main__':
    main()
