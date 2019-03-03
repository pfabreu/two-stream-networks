from keras.utils import to_categorical, multi_gpu_model
from keras import backend as K
import keras
import numpy as np
import utils as utils
from fusionUCF_model import TwoStreamModel
from fusionUCF_data_test import load_set, make_chunks
import pickle
import math
import gc
import scipy.io as spio
import os
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Erase previous models from GPU memory
    root_dir_info = "../../data/UCF101-24/"
    root_dir_data = "/media/pedro/actv-ssd/UCF101-24/"

    K.clear_session()

    sendmail = True
    attention_type = "crop"

    # Load list of action classes and separate them
    with open(root_dir_info + "classInd.txt") as f:
        classes = f.readlines()

    # Parameters for training
    params = {"dim": (224, 224), "batch_size": 32,
              "n_classes": len(classes), "n_channels_rgb": 3,
              "n_channels_of": 20, "of_length": 10,
              "shuffle": False, "nb_epochs": 50, "model": "resnet50",
              "train_chunk_size": 100}

    # Paths I need for data Load
    annot_path = root_dir_info + "pyannot2.pkl"
    index_path = "indexesDict2.pkl"
    testlist_path = root_dir_info + "testlist.pkl"
    with open(testlist_path, 'rb') as f:
        setlist = pickle.load(f)
    #trainsamples = len(setlist) * 0.75
    #trainlist = setlist[:math.floor(trainsamples)]
    #vallist = setlist[math.floor(trainsamples):]
    chunks = make_chunks(setlist, len(setlist), params["train_chunk_size"])
    #valchunks = make_chunks(vallist, len(vallist), params["train_chunk_size"])
    data_paths_dict = {"RGB": root_dir_data + "UCF_rgb/",
                       "crop": root_dir_data + "UCF_crop/",
                       "fovea": root_dir_data + "UCF_fovea/",
                       "gauss": root_dir_data + "UCF_gauss/",
                       "flow": root_dir_data + "tvl1_flow"}

    flow_saved_weights = "../models/flow_UCF.hdf5"  # "saved_models/rgb_fovea_resnet50_1806301953.hdf5"
    rgb_saved_weights = "../models/rgb_UCF_" + attention_type + ".hdf5"

    nsmodel = TwoStreamModel(params["n_classes"], rgb_saved_weights, flow_saved_weights)
    nsmodel.compile_model(soft_sigmoid=False)
    model = nsmodel.model
    model.load_weights("../models/UCF_fusion_" + attention_type + ".hdf5")

    chunks = make_chunks(setlist, len(setlist), params["train_chunk_size"])

    predictions = []
    X_list = []

    for chunk in chunks:
        # print(chunk)
        X_test_RGB = X_test_OF = y_test = None
        X_test_RGB, X_test_OF, y_test, data_list = load_set(annot_path, chunk, index_path, data_paths_dict["flow"], data_paths_dict[attention_type], params["dim"], params["n_channels_rgb"], params["n_channels_of"], params["of_length"])
        print("Loaded " + str(len(y_test)) + " data samples.\n")
        pred = model.predict([X_test_RGB, X_test_OF], batch_size=params['batch_size'], verbose=1)
        print(pred)
        predictions.append(pred)
        X_list.extend(data_list)

    predictions = np.vstack(predictions)

    class_predictions = {}
    score_predictions = {}
    with open(annot_path, 'rb') as f:
        annots = pickle.load(f)
    with open(testlist_path, 'rb') as f:
        testlist = pickle.load(f)
    with open(index_path, 'rb') as f:
        frame_indexes = pickle.load(f)

    x_pred_dict = {}
    counter = 0
    for entry in X_list:
        x_pred_dict[entry] = []
    for entry in X_list:
        x_pred_dict[entry].append(predictions[counter])
        counter += 1

    for key, data in x_pred_dict.iteritems():
        class_votes = np.zeros(24)
        score_votes = np.zeros(24)
        for pred in data:
            cls = pred.argmax(axis=0)  # Highest prediction
            class_votes[cls] += 1
            score_votes[cls] += pred[cls]
        c = class_votes.argmax(axis=0)  # Most voted
        classpred = c  # Because matlab
        scorepred = score_votes[c] / class_votes[c]  # Get average prediction
        class_predictions[key] = classpred
        score_predictions[key] = scorepred

    actions = ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving',
               'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing',
               'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing',
               'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping',
               'VolleyballSpiking', 'WalkingWithDog']

    f = open('testlist01.txt', 'r')
    testlist = f.readlines()
    f.close()
    print(len(testlist))

    data = spio.loadmat("../../data/UCF101-24/testlist.mat", struct_as_record=False, squeeze_me=True)
    testlist = data['testlist']

    keys = []
    tlist = []
    for st in testlist:
        keys.append(st)
        # keys.append(st.split(' ')[0][:-4])
        tlist.append(st + ".avi")

    # with open("testlist.pkl", 'wb') as f:
    #    pickle.dump(tlist, f)

    annot_path = "../../data/UCF101-24/pyannot2.pkl"
    with open(annot_path, 'rb') as f:
        annots = pickle.load(f)

    # dets_path = "../pred_crop.pkl"
    # with open(dets_path, 'rb') as f:
    #    detections = pickle.load(f)
    cm = np.zeros([24, 24], np.int32)
    pred_dict = []
    for key in keys:
        if key in annots:
            vid_name = key
            classes = []
            scores = []
            framenumbers = []
            boxes = []
            for BB in range(len(annots[key]['annotations'])):
                label = annots[key]['annotations'][BB]['label']
                tag = key + "_BB_" + str(BB)
                if tag in class_predictions:
                    # classes.append(label + 1)
                    classes.append(class_predictions[key + "_BB_" + str(BB)] + 1)
                    cm[label][class_predictions[key + "_BB_" + str(BB)]] += 1
                    cm[class_predictions[key + "_BB_" + str(BB)]][label] = cm[label][class_predictions[key + "_BB_" + str(BB)]]

                    bbs = annots[key]['annotations'][BB]['boxes']
                    # scores.append(1.0)
                    scores.append(score_predictions[key + "_BB_" + str(BB)])
                else:
                    print("Not found")
                    classes.append(label + 1)
                    scores.append(1.0)
                boxes.append({'bxs': bbs})
                ef = annots[key]['annotations'][BB]['ef']
                sf = annots[key]['annotations'][BB]['sf']
                framenumbers.append({'fnr': range(sf, ef)})
            pred_dict.append({'videoName': vid_name, 'class': classes, 'score': scores, 'framenr': framenumbers, 'boxes': boxes})
        else:
            print("This video is not in the annotations.")

    spio.savemat('predfusion_' + attention_type + '.mat', {'xmldata': pred_dict})
    g = sns.heatmap(cm, annot=True, fmt="d", xticklabels=actions, yticklabels=actions, linewidths=0.5, linecolor='black', cbar=True)
    plt.xticks(rotation=-90)
    plt.title("Confusion Matrix UCF101-24 " + attention_type)
    plt.show()

if __name__ == '__main__':
    main()
