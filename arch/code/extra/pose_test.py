import tensorflow as tf
import utils
import voting
from pose_model import pose_create_model, compile_model
from pose_data import load_split, get_AVA_set
import time
from keras import backend as K
import numpy as np
import pickle


def main():

    root_dir = '../../data/AVA/files/'

    # Load list of action classes and separate them (from utils_stream)
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    # Parameters for training
    params = {'dim': (300, 300), 'batch_size': 32,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'nb_epochs': 200, 'model': 'alexnet', 'email': True, 'train_chunk_size': 2**12,
              'validation_chunk_size': 2**12}
    soft_sigmoid = True
    store_predictions = True
    minValLoss = 9999990.0
    split = "test"

    # Get validation set from directory
    partition = {}
    partition['test'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_" + split.title() + "_Custom_Corrected.csv", soft_sigmoid=True)

    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    result_csv = "output_test_pose_" + time_str + ".csv"

    # Load trained model
    pose_weights = "../models/pose_alexnet_1808310209.hdf5"
    model = pose_create_model(classes=classes['label_id'], soft_sigmoid=soft_sigmoid, image_shape=params['dim'], model_name=params['model'])
    model = compile_model(model, soft_sigmoid=soft_sigmoid)
    model.load_weights(pose_weights)

    print("Test set size: " + str(len(partition['test'])))

    # Load chunks
    test_splits = utils.make_chunks(original_list=partition['test'], size=len(partition['test']), chunk_size=2**11)

    # Test directories where pre-processed test files are
    pose_dir = "/media/pedro/actv-ssd/pose_" + split + "/"

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

    test_predictions = []
    with tf.device('/gpu:0'):
        for testIDS in test_splits:
            # TODO Technically it shouldnt return labels here (these are ground truth)
            x_test_pose, y_test_pose, y_test_object, y_test_human = load_split(testIDS, None, params['dim'], params['n_channels'], split, filter_type, soft_sigmoid=True, train=False)
            print("Predicting on chunk " + str(test_chunks_count) + "/" + str(len(test_splits)))

            predictions = model.predict(x_test_pose, batch_size=params['batch_size'], verbose=1)
            if store_predictions is True:
                # print(predictions[0][0])
                # print(predictions[1][0])
                # print(predictions[2][0])

                # tarr = np.hstack((np.vstack(predictions[0]), np.vstack(predictions[1]), np.vstack(predictions[2])))
                test_predictions.append(predictions)

            # Convert predictions to readable output and perform majority voting
            voting.pred2classes(testIDS, predictions, pose_votes, obj_votes, human_votes, thresh=0.4)
            test_chunks_count += 1

    if store_predictions is True:
        #tp = np.vstack(test_predictions)
        # print(tp.shape)
        with open("thresholds/pose/predictions_pose_" + time_str + ".pickle", 'wb') as handle:
            pickle.dump(test_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    if params['email']:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com',
                        to_addr_list=['pedro_abreu95@hotmail.com'],
                        subject='Finished prediction for ' + filter_type,
                        message='Testing pose with following params: ' + str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
