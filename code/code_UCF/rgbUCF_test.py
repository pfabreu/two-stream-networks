from keras.utils import to_categorical, multi_gpu_model
from keras import backend as K
import keras
import utils as utils
from rgbUCF_model import rgb_create_model, compile_model
from rgbUCF_data import load_set
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as spio
import os
import numpy as np


def main():
    # Erase previous models from GPU memory
    root_dir_info = "../../data/UCF101-24/"
    root_dir_data = "/media/pedro/actv-ssd/UCF101-24/"

    K.clear_session()

    sendmail = False

    # Load list of action classes and separate them
    with open(root_dir_info + "classInd.txt") as f:
        classes = f.readlines()

    # Parameters for Testing
    params = {"dim": (224, 224), "batch_size": 32,
              "n_classes": len(classes), "n_channels": 3,
              "shuffle": False, "nb_epochs": 50, "model": "resnet50"}

    # Paths I need for data Load
    annot_path = root_dir_info + "pyannot2.pkl"
    testlist_path = root_dir_info + "testlist.pkl"

    data_paths_dict = {"RGB": root_dir_data + "UCF_rgb/",
                       "crop": root_dir_data + "UCF_crop/",
                       "fovea": root_dir_data + "UCF_fovea/",
                       "gauss": root_dir_data + "UCF_gauss/"}
    filter_type = "RGB"
    # Load X and Y for Testing set
    print("Now loading the Testing Set.\n")
    X_test, y_test = load_set(annot_path, testlist_path, data_paths_dict[filter_type], params["dim"], params["n_channels"])
    print(len(X_test))
    y_test = to_categorical(y_test, params["n_classes"])
    print("Loaded " + str(len(y_test)) + " data samples.\n")

    # Load X and Y for Val Set
    # Create + compile model, load saved weights if they exist
    model, keras_layer_names = rgb_create_model(params["n_classes"], model_name=params['model'], freeze_all=False, conv_fusion=False)
    model = compile_model(model, soft_sigmoid=False)

    model.load_weights("../models/rgb_UCF_" + filter_type + ".hdf5")

    # Start the Testing
    predictions = model.predict(X_test, batch_size=params['batch_size'], verbose=1)
    # print(predictions)

    # Voting
    class_predictions = {}
    score_predictions = {}
    with open(annot_path, 'rb') as f:
        annots = pickle.load(f)
    with open(testlist_path, 'rb') as f:
        testlist = pickle.load(f)
    keys = []
    counter = 0
    for st in testlist:
        keys.append(st.split(' ')[0][:-4])
    for key in keys:
        for BB in range(len(annots[key]['annotations'])):
            dirinfo = key.split('/')
            if data_paths_dict[filter_type] == "/media/pedro/actv-ssd/UCF101-24/UCF_rgb/":
                frame_path = dirinfo[1] + "/"
            else:
                frame_path = dirinfo[1] + "_BB_" + str(BB) + "/"
            class_votes = np.zeros(24)
            score_votes = np.zeros(24)
            for f in range(1, 6):
                if os.path.exists(data_paths_dict[filter_type] + frame_path + "frame" + str(f) + ".jpg"):
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

    if sendmail:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com',
                        to_addr_list=['pedro_abreu95@hotmail.com',
                                      'joaogamartins@gmail.com'],
                        cc_addr_list=[],
                        subject='Finished Testing UCF crop Resnet 50 RGB-stream ',
                        message='Testing RGB with following params: ' +
                        str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
