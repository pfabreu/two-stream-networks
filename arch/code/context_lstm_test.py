import os
CPU = True
if CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # This must be imported before keras

import voting
import pickle
import utils
import numpy as np
import time
from context_data import load_split, get_AVA_set, get_AVA_labels
from context_lstm_model import context_create_modelA, context_create_modelB
import pickle
from keras.callbacks import ModelCheckpoint
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

    # Load list of action classes and separate them (from utils_stream)
    root_dir = '../../data/AVA/files/'
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['test'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Test_Custom_Corrected.csv")  # IDs for training
    labels_test = get_AVA_labels(classes, partition, "test", filename=root_dir + "AVA_Test_Custom_Corrected.csv")

    # Create + compile model, load saved weights if they exist
    timewindow = 5  # NOTE To the past and to the future
    neighbours = 3
    num_classes = len(classes['label_name'])
    n_features = num_classes * neighbours
    context_dim = num_classes * neighbours * (timewindow + 1 + timewindow)
    modelname = "lstmA"
    NHU1V = [32, 64, 128, 256, 512]
    NHU2V = [16, 32, 64, 128, 256]
    for i in range(len(NHU1V)):
        NHU1 = NHU1V[i]
        NHU2 = NHU2V[i]

        bestModelPath = "../models/context/" + modelname + "/context_lstm_" + str(NHU1) + "_" + str(timewindow) + "_" + str(neighbours) + ".hdf5"
        if modelname == "lstmA":
            model = context_create_modelA(NHU1, NHU2, timewindow, n_features)
        elif modelname == "lstmB":
            model = context_create_modelB(NHU1, NHU2, timewindow, n_features)
        model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['categorical_accuracy'])
        model.load_weights(bestModelPath)

        time_str = time.strftime("%y%m%d%H%M", time.localtime())
        result_csv = "test_outputs/context/" + modelname + "/output_test_ctx_lstm_" + str(NHU1) + "_" + str(timewindow) + "_" + str(neighbours) + "_" + time_str + ".csv"

        pose_votes = {}
        obj_votes = {}
        human_votes = {}

        for row in partition['test']:
            row = row.split("@")
            i = row[0] + "@" + row[1] + "@" + str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
            pose_votes[i] = np.zeros(utils.POSE_CLASSES)
            obj_votes[i] = np.zeros(utils.OBJ_HUMAN_CLASSES)
            human_votes[i] = np.zeros(utils.HUMAN_HUMAN_CLASSES)

        x_test = y_test_pose = y_test_object = y_test_human = None
        Xfilename = root_dir + "context_files/XContext_test_tw" + str(timewindow) + "_n" + str(neighbours) + ".csv"
        x_test, y_test_pose, y_test_object, y_test_human = load_split(partition['test'], labels_test, context_dim, 1, "test", Xfilename)
        x_test_past, x_test_future = reshapeX(x_test, (timewindow + 1 + timewindow), n_features)
        predictions = model.predict([x_test_past, x_test_future], verbose=1)

        # Convert predictions to readable output and perform majority voting
        voting.pred2classes(partition['test'], predictions, pose_votes, obj_votes, human_votes, thresh=0.4)

        # When you're done getting all the votes, write output csv
        with open(result_csv, "a") as output_file:
            for key in pose_votes:
                idx = key.split("@")
                actions = []
                pv = pose_votes[key]
                pose_vote = pv.argmax(axis=0) + 1
                actions.append(pose_vote)

                # Get 3 top voted object
                ov = obj_votes[key]
                top_three_obj_votes = ov.argsort()[-3:][::-1] + utils.POSE_CLASSES + 1
                for t in top_three_obj_votes:
                    if t != 0:  # Often there might only be two top voted or one
                        actions.append(t)
                # Get 3 top voted human
                hv = human_votes[key]
                top_three_human_votes = hv.argsort()[-3:][::-1] + utils.POSE_CLASSES + utils.OBJ_HUMAN_CLASSES + 1
                for t in top_three_human_votes:
                    if t != 0:  # Often there might only be two top voted or one
                        actions.append(t)

                video_name = idx[0]
                timestamp = idx[1]
                bb_topx = idx[2]
                bb_topy = idx[3]
                bb_botx = idx[4]
                bb_boty = idx[5]
                for a in actions:
                    line = video_name + "," + timestamp + "," + bb_topx + "," + bb_topy + "," + bb_botx + "," + bb_boty + "," + str(a)
                    output_file.write("%s\n" % line)

if __name__ == '__main__':
    main()
