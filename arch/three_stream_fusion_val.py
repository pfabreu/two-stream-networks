import tensorflow as tf
import csv
import time

from three_stream_fusion_training_model import NStreamModel
from three_stream_fusion_test_data import get_AVA_set, get_AVA_classes, load_split

import numpy as np
from collections import Counter
import sys
from keras.utils import multi_gpu_model


def majorityVoting(voting_pose, voting_obj, voting_human):
    # Convert list of list (for obj and human) to list
    voting_obj = [item for sublist in voting_obj for item in sublist]
    voting_human = [item for sublist in voting_human for item in sublist]
    # Pick major vote among the arrays and write to the csv
    action_list = []
    obj_counter = Counter(voting_obj)
    human_counter = Counter(voting_human)
    pose_counter = Counter(voting_pose)
    pose_value, count = pose_counter.most_common()[0]  # Get the single most voted pose
    action_list.append(pose_value)
    o = obj_counter.most_common()[:3]  # If there are less than 3, it's no problem
    for tup in o:
        action_list.append(tup[0])  # tup[0] is value, tup[1] is count
    h = human_counter.most_common()[:3]
    for tup in h:
        action_list.append(tup[0])
    return action_list


def pred2classes(ids, predictions, output_csv):
    pose_list = []
    obj_list = []
    human_list = []

    OBJECT_THRESHOLD = 0.4
    HUMAN_THRESHOLD = 0.4

    i = 0
    for entry in predictions:
        for action_type in entry:
            arr = np.array(action_type)

            if i == 0:
                r = arr.argsort()[-1:][::-1]
                pose_list.append(r[0])
            elif i == 1:
                r = arr.argsort()[-3:][::-1]  # Get the three with the highest probabilities
                # TODO Get top 3 and check if they are above threshold
                p = r.tolist()
                prediction_list = []
                # print p
                for pred in p:
                    if arr[pred] > OBJECT_THRESHOLD:
                        # print arr[pred]
                        prediction_list.append(pred)
                # print prediction_list
                obj_list.append(prediction_list)
            elif i == 2:
                r = arr.argsort()[-3:][::-1]
                # TODO Get top 3 and check if they are above threshold
                p = r.tolist()
                # print p
                for pred in p:
                    if arr[pred] > HUMAN_THRESHOLD:
                        prediction_list.append(pred)
                        # print prediction_list
                human_list.append(prediction_list)
        i += 1
    voting_pose = []
    voting_obj = []
    voting_human = []
    i = 0

    with open(output_csv, "a") as output_file:
        for entry in ids:
            idx = entry.split("@")
            f = int(idx[6]) - 1
            voting_pose.append(pose_list[i])  # pose was a 1 element list
            voting_obj.append(obj_list[i])
            voting_human.append(human_list[i])
            if f == 4:
                action_list = majorityVoting(voting_pose, voting_obj, voting_human)
                # Write csv lines
                video_name = idx[0]
                timestamp = idx[1]
                bb_topx = idx[2]
                bb_topy = idx[3]
                bb_botx = idx[4]
                bb_boty = idx[5]

                for action in action_list:
                    line = video_name + "," + timestamp + "," + bb_topx + "," + bb_topy + "," + bb_botx + "," + bb_boty + "," + str(action)
                    output_file.write("%s\n" % line)
                # Reset the voting arrays
                voting_pose = []
                voting_obj = []
                voting_human = []

        i += 1


def main():

    # Load list of action classes and separate them (from utils_stream)
    classes = get_AVA_classes('../data/AVA/files/ava_action_list_v2.1.csv')

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['val'] = get_AVA_set(classes=classes, directory="/media/pedro/actv-ssd/foveated_val_gc/")  # IDs for training

    # Create + compile model, load saved weights if they exist
    # Create + compile model, load saved weights if they exist
    rgb_weights = "rgb_stream/models/rgb_resnet50_1805290059.hdf5"
    flow_weights = "flow_stream/models/flow_resnet50_1805290120.hdf5"
    context_weights = "context_stream/models/bestModelContext_256.hdf5"
    nsmodel = NStreamModel(classes['label_id'], rgb_weights, flow_weights, context_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model
    modelpath = "3stfusion_resnet50_1806060359.hdf5"  # Pick up where I left
    model.load_weights(modelpath)
    # Try to train on more than 1 GPU if possible
    # try:
    #    print("Trying MULTI-GPU")
    #    model = multi_gpu_model(model)
    # except:
    #    print("Multi-GPU failed")
    #    sys.exit(0)
    print("Val set size: " + str(len(partition['val'])))

    # Load first train_size of partition{'train'}
    val_chunk_size = 1025
    if val_chunk_size % 5 != 0:
        print(val_chunk_size + " has to be a multiple of 5")
        sys.exit(0)
    seq = partition['val']
    val_splits = [seq[i:i + val_chunk_size] for i in range(0, len(seq), val_chunk_size)]
    print("Validation splits: " + str(len(val_splits)))
    val_chunks_count = 0
    rgb_dir = "/media/pedro/actv-ssd/foveated_val_gc"
    flow_dir = "test/flow/actv-ssd/flow_val"
    Xfilename = "starter_list.csv"
    val_context_rows = {}
    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    output_csv = "output_3stream_val_" + time_str + ".csv"
    with open(Xfilename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            rkey = row[0] + "_" + row[1].lstrip("0") + \
                "@" + str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
            val_context_rows[rkey] = row[6]
    with tf.device('/gpu:0'):
        for valIDS in val_splits:
            x_rgb = x_flow = x_context = None
            x_rgb, x_flow, x_context = load_split(valIDS, (224, 224), 2, 10, rgb_dir, flow_dir, val_context_rows)
            predictions = model.predict([x_rgb, x_flow, x_context], batch_size=32, verbose=1)
            print("Val chunk " + str(val_chunks_count) + "/" + str(len(val_splits)))
            pred2classes(valIDS, predictions, output_csv)
            val_chunks_count += 1


if __name__ == '__main__':
    main()
