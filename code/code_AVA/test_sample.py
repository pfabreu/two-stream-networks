import csv
import tensorflow as tf
from keras import backend as K
import time
import timeit
import math
from rgb_data import load_split, get_AVA_set, get_AVA_labels
import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import random
import sys


def getID(row, sep, frame):
    video = row[0]
    kf_timestamp = row[1]
    # action = row[6]
    bb_top_x = row[2]
    bb_top_y = row[3]
    bb_bot_x = row[4]
    bb_bot_y = row[5]
    ID = video + sep + kf_timestamp.lstrip("0") + sep + str(bb_top_x) + sep + str(bb_top_y) + sep + str(bb_bot_x) + sep + str(bb_bot_y) + sep + str(frame)
    return ID


def oversampling(classes, root_dir, file):
    sep = "@"
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
            if math.ceil(avg_samples / types[i]) < 40.0:
                reps.append(int(math.ceil(avg_samples / types[i])) - 1)
            else:
                reps.append(int(40.0) - 1)
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
                    print("Index: " + str(index))
                    tag = row[0:5]
                    print(tag)

                    for m in range(index - 9, index + 9):
                        if m > 0 and m < l:
                            test_row = csv_list[m]
                            # Ger tag
                            test_tag = test_row[0:5]
                            # TODO for all rows that have the same bb in same vid and same time
                            if test_tag == tag:
                                for r in range(reps[cidx]):
                                    test_row[0] = "#" + test_row[0]  # NOTE This is needed as a signal for augmentation
                                    for frame in range(start_frame, end_frame + jump_frames, jump_frames):
                                        samples.append(getID(test_row, sep, frame))
                cidx += 1

                # Find all labels in AVA_Train_Custom_Corrected that correspond to each of these classes
    return samples, classes_to_rep


def undersampling(classes, root_dir, file, oversampling_classes):
    sep = "@"
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

    classes_to_remove = []
    removes = []

    # You can change avg_samples here
    no_undersampling = False
    beta = 2
    avg_samples = beta * avg_samples
    if no_undersampling:
        for i in range(len(types)):
            # TODO For debug, insert all classes
            removes.append(0)
            classes_to_remove.append(i + 1)
    else:
        for i in range(len(types)):
            if types[i] > avg_samples:
                removes.append(types[i] - avg_samples)
                classes_to_remove.append(i + 1)
    print(classes_to_remove)
    print(removes)

    g = sns.barplot(x=[str(i) for i in classes_to_remove], y=removes)
    plt.xticks(rotation=-90)
    plt.title(file + " samples to remove, with avg " + str(avg_samples))
    plt.grid(True)
    plt.show()
    samples = []
    removing_counter = np.zeros(len(removes))
    with open(root_dir + file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        # TODO Removal is removing the first N. Solution: Loads the rows first to another list and jumbles them up then processes that

        for row in csvReader:
            tp = int(row[6])
            tag = row[0:5]

            cidx = False
            for c in classes_to_remove:
                if tp == c:
                    cd = classes_to_remove.index(c)
                    if removing_counter[cd] < removes[cd]:
                        # video = "~" + row[0]
                        removing_counter[cd] += 1
                    else:
                        for frame in range(start_frame, end_frame + jump_frames, jump_frames):
                            # Append to the dictionary
                            samples.append(getID(row, sep, frame))
                    cidx = True
            if not cidx:
                for frame in range(start_frame, end_frame + jump_frames, jump_frames):
                    # Append to the dictionary
                    samples.append(getID(row, sep, frame))
    samples = list(set(samples))
    return samples, classes_to_remove


def main():

    GPU = False
    CPU = True
    num_cores = 8

    if GPU:
        num_GPU = 1
        num_CPU = 1
    if CPU:
        num_CPU = 1
        num_GPU = 0

    # config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
    #                        device_count={'CPU': num_CPU, 'GPU': num_GPU})
    # session = tf.Session(config=config)
    # K.set_session(session)

    # root_dir = '../../../AVA2.1/' # root_dir for the files
    root_dir = '../../data/AVA/files/'

    # Load list of action classes and separate them
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training
    params = {'dim': (224, 224), 'batch_size': 32,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 200, 'model': 'resnet50', 'email': True,
              'freeze_all': True, 'conv_fusion': False, 'train_chunk_size': 2**12,
              'validation_chunk_size': 2**12}
    soft_sigmoid = True
    minValLoss = 9999990.0
    print(classes)

    oversampling_train, oversampling_train_classes = oversampling(classes, root_dir, "AVA_Train_Custom_Corrected.csv")
    oversampling_val, oversampling_val_classes = oversampling(classes, root_dir, "AVA_Val_Custom_Corrected.csv")

    undersampling_train, undersampling_train_classes = undersampling(classes, root_dir, "AVA_Train_Custom_Corrected.csv", oversampling_train_classes)
    undersampling_val, undersampling_val_classes = undersampling(classes, root_dir, "AVA_Val_Custom_Corrected.csv", oversampling_val_classes)

    partition = {}
    partition['train'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Train_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)  # IDs for training
    partition['validation'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Val_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)  # IDs for validation

    print(len(partition['train']))
    undersampling_train = list(set(undersampling_train))
    print(len(oversampling_train))
    print(len(undersampling_train))
    print(len(undersampling_train + oversampling_train))
    print(1.0 * len(partition['train'] + oversampling_train) / len(partition['train']))
    bestsample = undersampling_train + oversampling_train

    sys.exit(0)
    # Labels
    # labels_train = get_AVA_labels(classes, partition, "train", filename=root_dir + "AVA_Train_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)
    # labels_val = get_AVA_labels(classes, partition, "validation", filename=root_dir + "AVA_Val_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)
    original_train_size = len(partition['train'])
    print("Training set size pre augmentation: " + str(original_train_size))
    partition['train'] = partition['train'] + aug_train
    print("Training set size pos augmentation: " + str(len(partition['train'])) + " --> " + str(100.0 * (len(partition['train']) - original_train_size) / original_train_size) + " % increase")

    original_val_size = len(partition['validation'])
    print("validation set size pre augmentation: " + str(original_train_size))
    partition['validation'] = partition['validation'] + aug_train
    print("Validation set size pos augmentation: " + str(len(partition['validation'])) + " --> " + str(100.0 * (len(partition['validation']) - original_val_size) / original_val_size) + " % increase")

    img = cv2.imread("/media/pedro/actv-ssd/gauss_train/-5KQ66BBWC4_902_0.077_0.151_0.283_0.811/frames1.jpg")
    print(img.shape)
    if random.random() < 0.5:
        flip_img = np.fliplr(img)
    crop_rand_val = random.randrange(0, 5, 1) / 10.0
    scale_rand_val = random.randrange(7, 8, 1) / 10.0
    # print(crop_rand_val)
    # print(scale_rand_val)
    seq = iaa.Sequential([  # horizontal flips
        iaa.Scale((scale_rand_val, 1.0)),
        iaa.CropAndPad(
            percent=(0, crop_rand_val),
            pad_mode=["edge"]
        )  # random crops
    ], random_order=True)  # apply augmenters in random order

    flipped = seq.augment_image(flip_img)
    plt.imshow(flipped)
    plt.show()
if __name__ == '__main__':
    main()
