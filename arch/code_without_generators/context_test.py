from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from context_model import context_create_model, compile_model
from context_data import load_split, get_AVA_set, get_AVA_labels
import voting

import pickle
import utils
import numpy as np
import time


def main():
    # Load list of action classes and separate them (from utils_stream)
    root_dir = '../../data/AVA/files/'
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': 270, 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 1,
              'shuffle': False, 'nb_epochs': 200, 'model': "mlp", 'email': True}

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['test'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Test_Custom_Corrected.csv")  # IDs for training

    # Labels
    labels_test = get_AVA_labels(classes, partition, "test", filename=root_dir + "AVA_Test_Custom_Corrected.csv")

    # Create + compile model, load saved weights if they exist
    bestModelPath = "../models/bestModelContext_" + str(128) + ".hdf5"
    model = context_create_model(128, 64, in_shape=(params['dim'],))
    model = compile_model(model)

    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    result_csv = "output_test_ctx_" + time_str + ".csv"

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
    x_test, y_test_pose, y_test_object, y_test_human = load_split(partition['test'], labels_test, params['dim'], params['n_channels'], "test", root_dir)

    predictions = model.predict(x_test, batch_size=params['batch_size'], verbose=1)

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
