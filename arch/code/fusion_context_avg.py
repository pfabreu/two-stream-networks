import os
CPU = False
if CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # This must be imported before keras

import tensorflow as tf
import utils
import voting
from two_stream_model import TwoStreamModel
from two_stream_data import get_AVA_set
from fusion_context_data import load_split
import time
from keras import backend as K
import numpy as np

import pickle
from context_data import get_AVA_labels
from context_lstm_model import context_create_modelA, context_create_modelB
from keras.callbacks import ModelCheckpoint
import itertools
from keras.layers import concatenate
import sys
import csv
from keras.utils import to_categorical
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
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
    # K.clear_session()

    # Load list of action classes and separate them (from utils_stream)
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': (224, 224), 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 150, 'model': 'resnet50', 'sendmail': True, 'gen_type': 'test'}

    # Get validation set from directory
    partition = {}
    partition['test'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Test_Custom_Corrected.csv", train=False)
    #partition['test'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Val_Custom_Corrected.csv", train=False)

    filter_type = "fovea"

    # Load trained model
    rgb_weights = "../models/rgb_fovea_resnet50_1806301953.hdf5"
    # rgb_weights = "../models/rgb_gauss_resnet50_1806290918.hdf5"
    # rgb_weights = "../models/rgb_crop_resnet50_1806300210.hdf5"
    #rgb_weights = "../models/rgb_rgb_resnet50_1807060914.hdf5"

    flow_weights = "../models/flow_resnet50_1806281901.hdf5"
    # flow_weights = "../models/flowcrop_resnet50_1807180022.hdf5"

    nsmodel = TwoStreamModel(classes['label_id'], rgb_weights, flow_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model
    # two_stream_weights = "../models/two_stream_fusion_elfovresnet50_1807030015.hdf5"
    two_stream_weights = "../models/two_stream_fusion_fovea_resnet50_1807030015.hdf5"
    model.load_weights(two_stream_weights)

    # Create + compile model, load saved weights if they exist
    timewindow = 5  # NOTE To the past and to the future
    neighbours = 3
    num_classes = len(classes['label_name'])
    n_features = num_classes * neighbours
    context_dim = num_classes * neighbours * (timewindow + 1 + timewindow)
    NHU1 = 512
    modelname = "lstmB"
    context_weights = "../models/context/" + modelname + "/context_lstm_" + str(NHU1) + "_" + str(timewindow) + "_" + str(neighbours) + ".hdf5"
    ctx_model = context_create_modelB(NHU1, NHU1 / 2, timewindow, n_features)
    ctx_model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['categorical_accuracy'])
    ctx_model.load_weights(context_weights)

    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    result_csv = "test_outputs/context_fusion/output_test_ctx_lstmavg_twophase_thresh01_" + str(NHU1) + "_" + str(timewindow) + "_" + str(neighbours) + "_" + time_str + ".csv"

    print("Test set size: " + str(len(partition['test'])))

    print("Building context dictionary from context file (these should be generated)...")
    #Xfilename = root_dir + "context_files/" + "XContext_val_tw" + str(timewindow) + "_n" + str(neighbours) + ".csv"
    #Xfilename = root_dir + "context_files/" + "XContext_test_tw" + str(timewindow) + "_n" + str(neighbours) + ".csv"
    Xfilename = root_dir + "context_files/" + "XContext_SecondPass_test_tw" + str(timewindow) + "_n" + str(neighbours) + ".csv"
    #Xfilename = root_dir + "context_files/" + "XContext_ThirdPass_test_tw" + str(timewindow) + "_n" + str(neighbours) + ".csv"
    test_context_rows = {}

    with open(Xfilename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            rkey = row[0] + "_" + row[1].lstrip("0") + "@" + str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
            test_context_rows[rkey] = row[6]

    # Test directories where pre-processed test files are
    # Load chunks
    test_splits = utils.make_chunks(original_list=partition['test'], size=len(partition['test']), chunk_size=2**10)

    rgb_dir = "/media/pedro/actv-ssd/" + filter_type + "_"
    flow_dir = "/media/pedro/actv-ssd/flow_"

    test_chunks_count = 0

    pose_votes = {}
    obj_votes = {}
    human_votes = {}

    for row in partition['test']:
        row = row.split("@")
        i = row[0] + "@" + row[1] + "@" + str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
        pose_votes[i] = np.zeros(utils.POSE_CLASSES)
        obj_votes[i] = np.zeros(utils.OBJ_HUMAN_CLASSES)
        human_votes[i] = np.zeros(utils.HUMAN_HUMAN_CLASSES)

    store_predictions = False
    test_predictions = []
    print("Starting testing:")
    with tf.device('/gpu:0'):
        for testIDS in test_splits:
            x_test_rgb, x_test_flow, x_test_context, y_test_pose, y_test_object, y_test_human = load_split(testIDS, None, params['dim'], params['n_channels'], 10, test_context_rows, rgb_dir, flow_dir, "rgb", params['gen_type'], train=False)

            print("Predicting on chunk " + str(test_chunks_count) + "/" + str(len(test_splits)) + ":")
            # Convert predictions to readable output and perform majority voting
            predictions_twostream = model.predict([x_test_rgb, x_test_flow], batch_size=params['batch_size'], verbose=1)
            x_test_past, x_test_future = reshapeX(x_test_context, (timewindow + 1 + timewindow), n_features)
            predictions_context = ctx_model.predict([x_test_past, x_test_future], batch_size=params['batch_size'], verbose=1)
            predictions = []
            print(len(predictions_twostream))
            for x1, x2 in zip(predictions_twostream, predictions_context):
                predictions.append((x1 + x2) / 2)

            if store_predictions is True:
                # print(predictions[0][0])
                # print(predictions[1][0])
                # print(predictions[2][0])

                # tarr = np.hstack((np.vstack(predictions[0]), np.vstack(predictions[1]), np.vstack(predictions[2])))
                test_predictions.append(predictions)
            voting.pred2classes(testIDS, predictions, pose_votes, obj_votes, human_votes, thresh=0.1)
            x_test_rgb = None
            x_test_flow = None
            x_test_context = None
            test_chunks_count += 1

    # When you're done getting all the votes, write output csv
    if store_predictions is True:
        #tp = np.vstack(test_predictions)
        # print(tp.shape)
        with open("thresholds/context_fusion/predictions_fusion_avg_" + filter_type + "_" + time_str + ".pickle", 'wb') as handle:
            pickle.dump(test_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    if params['sendmail']:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com',
                        to_addr_list=['pedro_abreu95@hotmail.com'],
                        subject='Finished ' + params['gen_type'] + ' prediction for three stream.',
                        message='Testing fusion with following params: ' + str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
