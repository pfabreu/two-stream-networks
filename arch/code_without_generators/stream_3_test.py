import tensorflow as tf
import csv
import time

from three_stream_fusion_training_model import ThreeStreamModel
from three_stream_fusion_test_data import get_AVA_set, get_AVA_classes, load_split

import sys


def main():

    # Load list of action classes and separate them (from utils_stream)
    classes = get_AVA_classes('../data/AVA/files/ava_action_list_v2.1.csv')

    root_dir = '../../data/AVA/files/'

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['test'] = get_AVA_set(classes=classes, directory="/media/pedro/actv-ssd/foveated_val_gc/")  # IDs for training

    # Create + compile model, load saved weights if they exist
    # Create + compile model, load saved weights if they exist
    rgb_weights = "rgb_stream/models/rgb_resnet50_1805290059.hdf5"
    flow_weights = "flow_stream/models/flow_resnet50_1805290120.hdf5"
    context_weights = "context_stream/models/bestModelContext_256.hdf5"
    nsmodel = ThreeStreamModel(classes['label_id'], rgb_weights, flow_weights, context_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model
    modelpath = "3stfusion_resnet50_1806060359.hdf5"  # Pick up where I left
    model.load_weights(modelpath)

    print("Testing set size: " + str(len(partition['val'])))

    # Load first train_size of partition{'train'}
    val_chunk_size = 1025
    if val_chunk_size % 5 != 0:
        print(val_chunk_size + " has to be a multiple of 5")
        sys.exit(0)
    seq = partition['val']
    test_splits = [seq[i:i + val_chunk_size] for i in range(0, len(seq), val_chunk_size)]
    print("Validation splits: " + str(len(val_splits)))
    val_chunks_count = 0
    rgb_dir = "/media/pedro/actv-ssd/foveated_val_gc"
    flow_dir = "test/flow/actv-ssd/flow_val"
    Xfilename = "starter_list.csv"
    val_context_rows = {}
    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    output_csv = "output_3stream_test_" + time_str + ".csv"
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
