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


def get_generators(classes, image_shape=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join('/data', 'train'),
        target_size=image_shape,
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join('/data', 'test'),
        target_size=image_shape,
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')

    return train_generator, validation_generator


# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'

#     def __init__(self, list_IDs, labels, gen_type="", data_augmentation=False, batch_size=64, dim=(224, 224), n_channels=3,
#                  n_classes=0, shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()

#         self.gen_type = gen_type
#         self.pre_resized = False
#         self.data_augmentation = False
#         self.dataset = 'AVA-split'

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         lower_bound = index * self.batch_size
#         upper_bound = (index + 1) * self.batch_size
#         indexes = self.indexes[lower_bound: upper_bound]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle is True:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples'
#         # Initialization, assuming its bidimensional (for now)
#         X = np.empty(
#             (self.batch_size, self.dim[0], self.dim[1], self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)

#         start_time = timeit.default_timer()
#         elapsed = 0
#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # print "g"
#             # Get image from ID (since we are using opencv we get np array)
#             split1 = ID.rsplit('_', 1)[0]
#             if self.pre_resized is True:
#                 img_name = "/media/pedro/actv4/" + self.dataset + "/" + self.gen_type + "_resize/" + \
#                     "/rgb/" + \
#                     split1.rsplit('_', 1)[0] + "/frame0000" + \
#                     ID.rsplit('_', 1)[1] + ".jpg"
#             else:
#                 img_name = "/media/pedro/actv4/" + self.dataset + "/" + self.gen_type + \
#                     "/rgb/" + \
#                     split1.rsplit('_', 1)[0] + "/frame0000" + \
#                     ID.rsplit('_', 1)[1] + ".jpg"
#             if not os.path.exists(img_name):
#                 print(img_name)
#                 print("[Error] File does not exist!")
#                 sys.exit(0)

#             img = cv2.imread(img_name)
#             if self.pre_resized is False:
#                 img = cv2.resize(
#                     img, (224, 224), interpolation=cv2.INTER_NEAREST)  # legends say pyrDown/pyrUp is actually faster
#                 # print "Resizing took: " + str(elapsed)
#             elapsed += timeit.default_timer() - start_time
#             # Store sample
#             X[i, ] = img

#             # Store class
#             y[i] = self.labels[ID]

#         # print elapsed
#         # print y.shape
#         # Return data and classes as 1-hot encoded
#         one_hot = keras.utils.to_categorical(y, num_classes=self.n_classes)
#         # print "One-hot shape" + str(one_hot.shape)
#         return X, one_hot
