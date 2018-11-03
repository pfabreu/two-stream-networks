from keras.utils import to_categorical, multi_gpu_model
from keras import backend as K
import keras
import numpy as np
import utils as utils
from flowUCF_model import flow_create_model, compile_model
from flowUCF_data import load_set, make_chunks
import pickle
import math
import scipy.io as spio
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Erase previous models from GPU memory
    root_dir_info = "../../data/UCF101-24/"
    root_dir_data = "/media/pedro/actv-ssd/UCF101-24/"

    K.clear_session()

    sendmail = True

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

    data_paths_dict = {"flow": root_dir_data + "tvl1_flow"}
    filter_type = "flow"

    model, keras_layer_names = flow_create_model(params["n_classes"], model_name=params['model'], freeze_all=False, conv_fusion=False)
    model = compile_model(model, soft_sigmoid=False)
    model.load_weights("../models/flow_UCF.hdf5")

    #trainsamples = len(setlist) * 0.75
    #trainlist = setlist[:math.floor(trainsamples)]
    #vallist = setlist[math.floor(trainsamples):]
    #chunks = make_chunks(setlist, len(setlist), params["train_chunk_size"])
    #valchunks = make_chunks(vallist, len(vallist), params["train_chunk_size"])
    chunks = make_chunks(setlist, len(setlist), params["train_chunk_size"])

    predictions = []
    for chunk in chunks:
        # print(chunk)
        X_test = y_test = None
        X_test, y_test = load_set(annot_path, chunk, index_path, data_paths_dict[filter_type], params["dim"], params["n_channels_of"], params["of_length"])
        print("Loaded " + str(len(y_test)) + " data samples.\n")
        pred = model.predict(X_test, batch_size=params['batch_size'], verbose=1)
        print(pred)
        predictions.append(pred)

    predictions = np.vstack(predictions)
    print(len(predictions))
    print(predictions[1])
    print(predictions[1000])
    # Start the TRAINING
    #bestModelPath = "saved_models/rgb_UCF_flow.hdf5"
    #saveBestModel = keras.callbacks.ModelCheckpoint(bestModelPath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    #maxValAcc = 0
    # for epoch in range(params["nb_epochs"]):
    #    print("Now on epoch " + str(epoch) + ".\n")
    #    epochAccuracies = []
    #    for chunk in chunks:
    #        X_Train = y_Train = X_Val = y_Val = None
    #        X_Train, y_Train = load_set(annot_path, chunk, index_path, data_paths_dict["flow"], params["dim"], params["n_channels_of"], params["of_length"])
    #        y_Train = to_categorical(y_Train, params["n_classes"])
    #        history = model.fit(X_Train, y_Train, batch_size=params['batch_size'], epochs=1, verbose=1, shuffle=True)
    #        gc.collect()
    #    for chunk in valchunks:
    #        X_Train = y_Train = X_Val = y_Val = None
    #        X_Val, y_Val = load_set(annot_path, chunk, index_path, data_paths_dict["flow"], params["dim"], params["n_channels_of"], params["of_length"])
    #        y_Val = to_categorical(y_Val, params["n_classes"])
    #        loss, acc = model.evaluate(X_Val, y_Val, batch_size=params["batch_size"], verbose=0)
    #        epochAccuracies.append(acc)
    #    currentValAcc = sum(epochAccuracies) / len(epochAccuracies)
    #    if currentValAcc > maxValAcc:
    #        maxValAcc = currentValAcc
    #        model.save(bestModelPath)
    #        print("New best model. Best val_acc is now " + str(maxValAcc) + ".\n")

    class_predictions = {}
    score_predictions = {}
    with open(annot_path, 'rb') as f:
        annots = pickle.load(f)
    with open(testlist_path, 'rb') as f:
        testlist = pickle.load(f)
    with open(index_path, 'rb') as f:
        frame_indexes = pickle.load(f)
    keys = []
    counter = 0
    for st in testlist:
        keys.append(st.split(' ')[0][:-4])
    for key in keys:
        for BB in range(len(annots[key]['annotations'])):
            dirinfo = key.split('/')
            index_key = dirinfo[1] + "_BB_" + str(BB)
            current_indexes = frame_indexes[index_key]
            class_votes = np.zeros(24)
            score_votes = np.zeros(24)
            # print(current_indexes)
            if current_indexes:
                for img_number in current_indexes:
                    pred = predictions[counter]
                    cls = pred.argmax(axis=0)  # Highest prediction
                    class_votes[cls] += 1
                    score_votes[cls] += pred[cls]
                    counter += 1
            c = class_votes.argmax(axis=0)  # Most voted
            classpred = c  # Because matlab
            scorepred = score_votes[c] / class_votes[c]  # Get average prediction
            class_predictions[key + "_BB_" + str(BB)] = classpred
            score_predictions[key + "_BB_" + str(BB)] = scorepred
    print(counter)

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
                # classes.append(label + 1)
                classes.append(class_predictions[key + "_BB_" + str(BB)] + 1)
                cm[label][class_predictions[key + "_BB_" + str(BB)]] += 1
                cm[class_predictions[key + "_BB_" + str(BB)]][label] = cm[label][class_predictions[key + "_BB_" + str(BB)]]

                bbs = annots[key]['annotations'][BB]['boxes']
                # scores.append(1.0)
                scores.append(score_predictions[key + "_BB_" + str(BB)])

                boxes.append({'bxs': bbs})
                ef = annots[key]['annotations'][BB]['ef']
                sf = annots[key]['annotations'][BB]['sf']
                framenumbers.append({'fnr': range(sf, ef)})
            pred_dict.append({'videoName': vid_name, 'class': classes, 'score': scores, 'framenr': framenumbers, 'boxes': boxes})
        else:
            print("This video is not in the annotations.")

    spio.savemat('pred_' + filter_type + '.mat', {'xmldata': pred_dict})
    g = sns.heatmap(cm, annot=True, fmt="d", xticklabels=actions, yticklabels=actions, linewidths=0.5, linecolor='black', cbar=True)
    plt.xticks(rotation=-90)
    plt.title("Confusion Matrix UCF101-24 " + filter_type)
    plt.show()

if __name__ == '__main__':
    main()
