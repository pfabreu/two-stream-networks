import os
CPU = True
if CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # This must be imported before keras

from context_data import load_split, get_AVA_set, get_AVA_labels
from context_lstm_model import context_create_modelA, context_create_modelB
import pickle
import numpy as np
from keras.callbacks import ModelCheckpoint
import utils
import itertools
from keras.layers import concatenate
import sys
from keras.utils import to_categorical
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from keras import backend as K


def reshapeX(x_train, timesteps, features):
    ln = x_train.shape[0]
    i = 0
    print(ln)
    X_past = np.zeros([ln, (timesteps // 2) + 1, features])
    X_future = np.zeros([ln, (timesteps // 2) + 1, features])
    for xline in x_train:
        xline = np.split(xline, timesteps)
        t = 0
        for xtime in xline[:((timesteps // 2) + 1)]:
            X_past[i, t, ] = xtime
            t += 1

        t = 0
        for xtime in xline[(timesteps // 2):]:
            X_past[i, t, ] = xtime
            t += 1
        i += 1
    print(X_past.shape)
    print(X_future.shape)
    return X_past, X_future


def main():
    K.clear_session()

    root_dir = '../../data/AVA/files/'
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    partition = {}
    timewindows = [1]
    # neighbs = [1, 2, 3]
    # Labels
    # timewindow = 10  # NOTE To the past and to the future
    neighbours = 3
    num_classes = len(classes['label_name'])
    n_features = num_classes * neighbours

    for timewindow in timewindows:
        context_dim = num_classes * neighbours * (timewindow + 1 + timewindow)
        Xfilename = root_dir + "context_files/XContext_train_tw" + str(timewindow) + "_n" + str(neighbours) + ".csv"
        partition['train'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Train_Custom_Corrected.csv")  # IDs for training
        labels_train = get_AVA_labels(classes, partition, "train", filename=root_dir + "AVA_Train_Custom_Corrected.csv")
        x_train, y_train_pose, y_train_object, y_train_human = load_split(partition['train'], labels_train, context_dim, 1, "train", Xfilename)
        y_t = []
        y_t.append(to_categorical(y_train_pose, num_classes=utils.POSE_CLASSES))
        y_t.append(utils.to_binary_vector(y_train_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
        y_t.append(utils.to_binary_vector(y_train_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))
        x_train_past, x_train_future = reshapeX(x_train, (timewindow + 1 + timewindow), n_features)
        # y_train = reshapeY(y_t, num_classes)

        # Load val data
        Xfilename = root_dir + "context_files/XContext_val_tw" + str(timewindow) + "_n" + str(neighbours) + ".csv"
        partition['validation'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Val_Custom_Corrected.csv")  # IDs for training
        labels_val = get_AVA_labels(classes, partition, "validation", filename=root_dir + "AVA_Val_Custom_Corrected.csv")
        x_val, y_val_pose, y_val_object, y_val_human = load_split(partition['validation'], labels_val, context_dim, 1, "validation", Xfilename)
        y_v = []
        y_v.append(to_categorical(y_val_pose, num_classes=utils.POSE_CLASSES))
        y_v.append(utils.to_binary_vector(y_val_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
        y_v.append(utils.to_binary_vector(y_val_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))
        x_val_past, x_val_future = reshapeX(x_val, (timewindow + 1 + timewindow), n_features)
        # y_val = reshapeY(y_v, num_classes)

        # NOTE This is a hack, training on the actual validation (but chunk will be removed)
        x_train_past = np.vstack((x_train_past, x_val_past))
        x_train_future = np.vstack((x_train_future, x_val_future))
        y_t[0] = np.vstack((y_t[0], y_v[0]))
        y_t[1] = np.vstack((y_t[1], y_v[1]))
        y_t[2] = np.vstack((y_t[2], y_v[2]))

        modelname = "lstmA"
        # y_train = np.vstack((y_train, y_val))
        #NHU1V = [32, 64, 128, 256, 512]
        #NHU2V = [16, 32, 64, 128, 256]
        NHU1V = [1024]
        NHU2V = [512]
        for i in range(len(NHU1V)):
            NHU1 = NHU1V[i]
            NHU2 = NHU2V[i]

            print("Now on Model with " + str(NHU1) + "," + str(NHU2) + " " + str(timewindow) + "\n")
            bestModelPath = "../models/context/" + modelname + "/context_lstm_" + str(NHU1) + "_" + str(timewindow) + "_" + str(neighbours) + ".hdf5"
            histPath = "../loss_acc_plots/" + modelname + "/results_LSTM_" + str(NHU1) + "_" + str(timewindow) + "_" + str(neighbours)
            checkpointer = ModelCheckpoint(filepath=bestModelPath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, period=1)

            # Create LSTM
            if modelname == "lstmA":
                model = context_create_modelA(NHU1, NHU2, timewindow, n_features)
            elif modelname == "lstmB":
                model = context_create_modelB(NHU1, NHU2, timewindow, n_features)
            # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['categorical_accuracy'])

            # Train
            n_epoch = 100
            hist = model.fit([x_train_past, x_train_future], y_t, shuffle=True, validation_split=0.1, epochs=n_epoch, verbose=0, callbacks=[checkpointer])

            # model.save(bestModelPath)
            with open(histPath, 'wb') as file_pi:
                pickle.dump(hist.history, file_pi)


if __name__ == '__main__':
    main()
