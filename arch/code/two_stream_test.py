import tensorflow as tf
import utils
import voting
from two_stream_model import TwoStreamModel
from two_stream_data import get_AVA_set, load_split
import time
from keras import backend as K
import numpy as np


def main():
    GPU = False
    CPU = True
    num_cores = 8

    if GPU:
        num_GPU = 1
        num_CPU = 1
    if CPU:
        num_CPU = 1
        num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)

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

    filter_type = "gauss"

    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    result_csv = "test_outputs/two-streams/output_" + params['gen_type'] + "_2stream_" + filter_type + "_" + time_str + ".csv"

    # Load trained model
    # rgb_weights = "../models/rgb_fovea_resnet50_1806301953.hdf5"
    rgb_weights = "../models/rgb_gauss_resnet50_1806290918.hdf5"
    # rgb_weights = "../models/rgb_crop_resnet50_1806300210.hdf5"

    flow_weights = "../models/flow_resnet50_1806281901.hdf5"

    nsmodel = TwoStreamModel(classes['label_id'], rgb_weights, flow_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model
    # two_stream_weights = "../models/two_stream_fusion_elfovresnet50_1807030015.hdf5"
    two_stream_weights = "../models/two_stream_fusion_gaussian_resnet50_1807121121.hdf5"
    model.load_weights(two_stream_weights)

    print("Test set size: " + str(len(partition['test'])))

    # Load chunks
    test_splits = utils.make_chunks(original_list=partition['test'], size=len(partition['test']), chunk_size=2**10)

    # Test directories where pre-processed test files are

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

    with tf.device('/gpu:0'):
        for testIDS in test_splits:
            x_test_rgb, x_test_flow, y_test_pose, y_test_object, y_test_human = load_split(testIDS, None, params['dim'], params['n_channels'], 10, rgb_dir, flow_dir, "test", "rgb", train=False)
            print("Predicting on chunk " + str(test_chunks_count) + "/" + str(len(test_splits)) + ":")
            # Convert predictions to readable output and perform majority voting
            predictions = model.predict([x_test_rgb, x_test_flow], batch_size=params['batch_size'], verbose=1)
            voting.pred2classes(testIDS, predictions, pose_votes, obj_votes, human_votes, thresh=0.4)
            x_test_rgb = None
            x_test_flow = None
            test_chunks_count += 1

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

    if params['sendmail']:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com',
                        to_addr_list=['pedro_abreu95@hotmail.com'],
                        subject='Finished ' + params['gen_type'] + ' prediction for two stream.',
                        message='Testing fusion with following params: ' + str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
