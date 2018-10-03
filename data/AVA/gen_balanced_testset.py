import tensorflow as tf
# from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras import backend as K
import csv
import time
import timeit
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from itertools import islice


def writeCSV(in_line):

    return


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
            bb_top_x = row[2]
            bb_top_y = row[3]
            bb_bot_x = row[4]
            bb_bot_y = row[5]
            a = row[6]
            ID = video + sep + kf_timestamp.lstrip("0") + sep + str(bb_top_x) + sep + str(bb_top_y) + sep + str(bb_bot_x) + sep + str(bb_bot_y) + sep + str(a)
            id_list.append(ID)
    id_list = list(set(id_list))  # Make sure we only got unique id's
    return id_list


def oversampling(classes, root_dir, file):
    sep = "@"
    repeating_threshold = 60.0
    start_frame = 1
    end_frame = 5
    jump_frames = 1  # Keyframe will be 3
    types = np.zeros(len(classes['label_id']))

    avg_samples = 0
    with open(root_dir + file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            tp = int(row[6]) - 1
            types[tp] += 1
            avg_samples += 1
    avg_samples /= len(classes['label_id'])

    print(types)
    print(avg_samples)
    classes_to_rep = []
    reps = []
    for i in range(len(types)):
        if types[i] < avg_samples and types[i] != 0:
            # print("Class: " + str(i + 1))
            # print("Samples: " + str(types[i]))
            # print("Reps: " + str(math.ceil(avg_samples / types[i])))
            if math.ceil(avg_samples / types[i]) < repeating_threshold:
                reps.append(int(math.ceil(avg_samples / types[i])) - 1)
            else:
                reps.append(int(repeating_threshold) - 1)
            classes_to_rep.append(i + 1)
    print(classes_to_rep)
    print(reps)

    g = sns.barplot(x=[str(i) for i in classes_to_rep], y=reps)
    plt.xticks(rotation=-90)
    plt.title(file + " reps, with avg " + str(avg_samples))
    plt.grid(True)
    plt.show()
    # TODO Histogram to show how many reps per class
    samples = []
    with open(root_dir + file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        csv_list = list(csvReader)
        l = len(csv_list)
        for index, row in enumerate(csv_list):
            tp = int(row[6])
            cidx = 0
            for c in classes_to_rep:
                if tp == c:
                    #print("Index: " + str(index))
                    tag = row[0:5]
                    # print(tag)

                    for m in range(index - 7, index + 7):
                        if m > 0 and m < l:
                            test_row = csv_list[m]
                            if test_row[0][0] == "#":
                                # print(test_row)
                                sys.exit(0)
                            # Ger tag
                            test_tag = test_row[0:5]
                            # TODO for all rows that have the same bb in same vid and same time
                            if test_tag == tag:

                                for r in range(reps[cidx]):
                                    # NOTE This is needed as a signal for augmentation
                                    # for frame in range(start_frame, end_frame + jump_frames, jump_frames):
                                    video = test_row[0]
                                    kf_timestamp = test_row[1]
                                    # action = row[6]
                                    bb_top_x = test_row[2]
                                    bb_top_y = test_row[3]
                                    bb_bot_x = test_row[4]
                                    bb_bot_y = test_row[5]
                                    a = test_row[6]
                                    ID = video + sep + kf_timestamp.lstrip("0") + sep + str(bb_top_x) + sep + str(bb_top_y) + sep + str(bb_bot_x) + sep + str(bb_bot_y) + sep + str(a)
                                    samples.append(ID)
                                test_row[0] = row[0]
                cidx += 1

                # Find all labels in AVA_Train_Custom_Corrected that correspond to each of these classes
    return samples, classes_to_rep


def main():
    # root_dir = '../../../AVA2.1/' # root_dir for the files
    root_dir = 'files/'

    # Load list of action classes and separate them
    classes = get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    soft_sigmoid = True

    # TODO Oversampling
    aug_test, aug_test_classes = oversampling(classes, root_dir, "AVA_Test_Custom_Corrected.csv")

    #  TODO Undersampling
    # undersampling_train, undersampling_train_classes = undersampling(classes, root_dir, "AVA_Train_Custom_Corrected.csv", oversampling_train_classes)
    # undersampling_val, undersampling_val_classes = undersampling(classes, root_dir, "AVA_Val_Custom_Corrected.csv", oversampling_val_classes)

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['test'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Test_Custom_Corrected.csv")  # IDs for training
    partition['test'] = partition['test'] + aug_test
    with open(filename, 'a') as f:

        for i in partition['test']:
            row = i.split("@")
            video = row[0]
            kf_timestamp = row[1]
            # action = row[6]
            bb_top_x = row[2]
            bb_top_y = row[3]
            bb_bot_x = row[4]
            bb_bot_y = row[5]
            a = row[6]
            csvLine = videoName + ',' + keyFrame + ',' + str(bb[0]) + ',' + str(bb[1]) + ',' + str(bb[2]) + ',' + str(bb[3]) + ',' + str(action) + '\n'

            f.write(csvLine)

    # TODO Write CSV

if __name__ == '__main__':
    main()
