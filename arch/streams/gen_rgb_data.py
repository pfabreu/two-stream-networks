"""
Class for managing our data.
"""
import csv
import numpy as np
import cv2
import os.path
import keras
import sys
import timeit


def load_split(ids, labels, dim, n_channels, gen_type):
    'Generates data containing batch_size samples'
    # Initialization, assuming its bidimensional (for now)
    X = np.empty([len(ids), dim[0], dim[1], n_channels])
    y = np.empty(len(ids))
    # Generate data
    for i, ID in enumerate(ids):
        # print "g"
        # Get image from ID (since we are using opencv we get np array)
        split1 = ID.rsplit('_', 1)[0]
        img_name = "ava_" + gen_type + "_resized/rgb/" + split1.rsplit('_', 1)[0] + "/frame0000" + \
            ID.rsplit('_', 1)[1] + ".jpg"
        if not os.path.exists(img_name):
            print(img_name)
            print("[Error] File does not exist!")
            sys.exit(0)

        img = cv2.imread(img_name)
        #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        # Store sample
        X[i, ] = img
        y[i] = labels[ID]

    # print elapsed
    # print y.shape
    # print "One-hot shape" + str(one_hot.shape)
    return X, y


# AHA data load functions


# AVA data load functions
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


def get_AVA_set(classes, filename, dir):
    # Load all lines of filename
    id_list = []
    start_frame = 25
    end_frame = 65
    jump_frames = 10  # Keyframe will be 45
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            video = row[0]
            kf_timestamp = row[1]
            action = row[6]
            # This is due to the behav of range
            for frame in range(start_frame, end_frame +
                               jump_frames, jump_frames):
                # Append to the dictionary
                ID = video + "_" + kf_timestamp.lstrip("0") + \
                    "_" + action + "_" + str(frame)

                id_list.append(ID)
    return id_list


def get_AVA_labels(classes, partition, set_type):
    labels = {}
    # Parse partition and create a correspondence to an integer in classes
    class_ids = classes['label_id']
    # First process the training
    for entry in partition[set_type]:
        labels[entry] = int(entry.split('_')[-2]) - 1
        # print labels[entry]
    # Then the validation
    # for entry in partition['validation']:
    #    labels[entry] = int(entry.split('_')[-2])-1
        # print labels[entry]
    return labels


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, gen_type="", data_augmentation=False, batch_size=64, dim=(224, 224), n_channels=3,
                 n_classes=0, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        self.gen_type = gen_type
        self.pre_resized = False
        self.data_augmentation = False
        self.dataset = 'AVA-split'

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        lower_bound = index * self.batch_size
        upper_bound = (index + 1) * self.batch_size
        indexes = self.indexes[lower_bound: upper_bound]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization, assuming its bidimensional (for now)
        X = np.empty(
            (self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        start_time = timeit.default_timer()
        elapsed = 0
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # print "g"
            # Get image from ID (since we are using opencv we get np array)
            split1 = ID.rsplit('_', 1)[0]
            if self.pre_resized is True:
                img_name = "/media/pedro/actv4/" + self.dataset + "/" + self.gen_type + "_resize/" + \
                    "/rgb/" + \
                    split1.rsplit('_', 1)[0] + "/frame0000" + \
                    ID.rsplit('_', 1)[1] + ".jpg"
            else:
                img_name = "/media/pedro/actv4/" + self.dataset + "/" + self.gen_type + \
                    "/rgb/" + \
                    split1.rsplit('_', 1)[0] + "/frame0000" + \
                    ID.rsplit('_', 1)[1] + ".jpg"
            if not os.path.exists(img_name):
                print(img_name)
                print("[Error] File does not exist!")
                sys.exit(0)

            img = cv2.imread(img_name)
            if self.pre_resized is False:
                img = cv2.resize(
                    img, (224, 224), interpolation=cv2.INTER_NEAREST)  # legends say pyrDown/pyrUp is actually faster
                # print "Resizing took: " + str(elapsed)
            elapsed += timeit.default_timer() - start_time
            # Store sample
            X[i, ] = img

            # Store class
            y[i] = self.labels[ID]

        # print elapsed
        # print y.shape
        # Return data and classes as 1-hot encoded
        one_hot = keras.utils.to_categorical(y, num_classes=self.n_classes)
        # print "One-hot shape" + str(one_hot.shape)
        return X, one_hot
