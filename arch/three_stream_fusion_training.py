# Learn merged rgb-stream filters (visualize them)
import tensorflow as tf
import numpy as np
from utils.utils import *
from keras.utils import to_categorical
from three_stream_fusion_training_model import NStreamModel
from three_stream_fusion_training_data import get_AVA_set, get_AVA_labels, get_AVA_classes, load_split
import time
import csv
import timeit


def to_binary_vector(list_classes, size, labeltype):
    """
    Keras should have this function...
    """
    labelsarray = np.empty([len(list_classes), size])
    offset = 0
    if labeltype == 'object-human':
        offset = POSE_CLASSES
    elif labeltype == 'human-human':
        offset = POSE_CLASSES + OBJ_HUMAN_CLASSES
    elif labeltype == 'pose':
        offset = 0
    index = 0
    for l in list_classes:
        bv = np.zeros(size)
        lv = l
        if len(lv) >= 1:
            lv = [x - offset for x in lv]
        for c in lv:
            v = to_categorical(c, size)
            bv = bv + v

        labelsarray[index, :] = bv
        index += 1
    return labelsarray


def main():
    sendmail = True

    # Load list of action classes and separate them (from _stream)
    classes = get_AVA_classes('../data/AVA/files/ava_action_list_v2.1.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': (224, 224), 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 50, 'model': 'resnet50'}

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['train'] = get_AVA_set(
        classes=classes, filename="../data/AVA/files/ava_mini_split_train_big.csv")  # IDs for training
    partition['train'] = list(set(partition['train']))

    partition['validation'] = get_AVA_set(
        classes=classes, filename="../data/AVA/files/ava_mini_split_val_big.csv")  # IDs for validation
    partition['validation'] = list(set(partition['validation']))

    # Labels
    labels_train = get_AVA_labels(classes, partition, "train", filename="../data/AVA/files/ava_mini_split_train_big.csv")
    labels_val = get_AVA_labels(classes, partition, "validation", filename="../data/AVA/files/ava_mini_split_val_big.csv")

    # Create + compile model, load saved weights if they exist
    rgb_weights = "rgb_stream/models/rgb_resnet50_1805290059.hdf5"
    flow_weights = "flow_stream/models/flow_resnet50_1805290120.hdf5"
    context_weights = "context_stream/models/bestModelContext_256.hdf5"
    nsmodel = NStreamModel(classes['label_id'], rgb_weights, flow_weights, context_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model
    modelpath = "3stfusion_resnet50_1806060359.hdf5"  # Pick up where I left
    if modelpath is not None:
        print("Loading previous weights")
        model.load_weights(modelpath)

    print("Training set size: " + str(len(partition['train'])))

    # Load first train_size of partition{'train'}
    train_size = 2**15  # 65536
    seq = partition['train'][:train_size]
    train_chunk_size = 2**10  # 4096
    val_chunk_size = 2**10
    train_splits = [seq[i:i + train_chunk_size] for i in range(0, len(seq), train_chunk_size)]
    val_size = 2**12
    seq_val = partition['validation'][:val_size]
    num_val_chunks = 1  # Let's say we want 10 val chunks
    val_splits = [seq_val[i:i + val_chunk_size] for i in range(0, len(seq_val), val_chunk_size)]

    maxValAcc = 0.0
    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    bestModelPath = "3stfusion_" + params['model'] + "_" + time_str + ".hdf5"
    traincsvPath = "3stfusion_train_plot_" + params['model'] + "_" + time_str + ".csv"
    valcsvPath = "3stfusion_val_plot_" + params['model'] + "_" + time_str + ".csv"

    print("Building context dictionary...")
    Xfilename = "context_stream/files/XContext_train_pastfuture.csv"
    train_context_rows = {}
    with open(Xfilename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            rkey = row[0] + "_" + row[1].lstrip("0") + \
                "@" + str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
            train_context_rows[rkey] = row[6]

    Xfilename = "context_stream/files/XContext_val_pastfuture.csv"
    val_context_rows = {}
    with open(Xfilename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            rkey = row[0] + "_" + row[1].lstrip("0") + \
                "@" + str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
            val_context_rows[rkey] = row[6]
    print("Finished building context dictionary...")

    with tf.device('/gpu:0'):
        for epoch in range(params['nb_epochs']):
            epoch_chunks_count = 0
            for trainIDS in train_splits:
                start_time = timeit.default_timer()
                # -----------------------------------------------------------
                x_val_rgb = x_val_flow = x_val_context = y_val_pose = y_val_object = y_val_human = x_train_rgb = x_train_flow = y_train_pose = y_train_object = y_train_human = None
                x_train_rgb, x_train_flow, x_train_context, y_train_pose, y_train_object, y_train_human = load_split(trainIDS, labels_train, params['dim'], params['n_channels'], "train", 10, train_context_rows)

                y_train_pose = to_categorical(y_train_pose, num_classes=POSE_CLASSES)
                y_train_object = to_binary_vector(y_train_object, size=OBJ_HUMAN_CLASSES, labeltype='object-human')
                y_train_human = to_binary_vector(y_train_human, size=HUMAN_HUMAN_CLASSES, labeltype='human-human')

                history = model.fit([x_train_rgb, x_train_flow, x_train_context], [y_train_pose, y_train_object, y_train_human], batch_size=params['batch_size'], epochs=1, verbose=0)
                elapsed = timeit.default_timer() - start_time
                # learning_rate_schedule(model, epoch, params['nb_epochs'])
                # ------------------------------------------------------------
                print("Epoch " + str(epoch) + " chunk " + str(epoch_chunks_count) + " (" + str(elapsed) + ") acc[pose,obj,human] = [" + str(history.history['pred_pose_categorical_accuracy']) + "," +
                      str(history.history['pred_obj_human_categorical_accuracy']) + "," + str(history.history['pred_human_human_categorical_accuracy']) + "] loss: " + str(history.history['loss']))
                with open(traincsvPath, 'a') as f:
                    writer = csv.writer(f)
                    avg_acc = (history.history['pred_pose_categorical_accuracy'][0] + history.history['pred_obj_human_categorical_accuracy'][0] + history.history['pred_human_human_categorical_accuracy'][0]) / 3
                    writer.writerow([str(avg_acc), history.history['pred_pose_categorical_accuracy'], history.history['pred_obj_human_categorical_accuracy'], history.history['pred_human_human_categorical_accuracy'], history.history['loss']])

                epoch_chunks_count += 1
            print("Validating data: ")
            global_loss = 0.0
            pose_loss = 0.0
            object_loss = 0.0
            human_loss = 0.0
            pose_acc = 0.0
            object_acc = 0.0
            human_acc = 0.0
            for val_idx in range(num_val_chunks):
                x_val_rgb = x_val_flow = x_val_context = y_val_pose = y_val_object = y_val_human = x_train_rgb = x_train_flow = y_train_pose = y_train_object = y_train_human = None
                x_val_rgb, x_val_flow, x_val_context, y_val_pose, y_val_object, y_val_human = load_split(val_splits[val_idx][:val_chunk_size], labels_val, params['dim'], params['n_channels'], "val", 10, val_context_rows)

                y_val_pose = to_categorical(y_val_pose, num_classes=POSE_CLASSES)
                y_val_object = to_binary_vector(y_val_object, size=OBJ_HUMAN_CLASSES, labeltype='object-human')
                y_val_human = to_binary_vector(y_val_human, size=HUMAN_HUMAN_CLASSES, labeltype='human-human')

                vglobal_loss, vpose_loss, vobject_loss, vhuman_loss, vpose_acc, vobject_acc, vhuman_acc = model.evaluate([x_val_rgb, x_val_flow, x_val_context], [y_val_pose, y_val_object, y_val_human], batch_size=params['batch_size'])
                global_loss += vglobal_loss
                pose_loss += vpose_loss
                object_loss += vobject_loss
                human_loss += vhuman_loss
                pose_acc += vpose_acc
                object_acc += vobject_acc
                human_acc += vhuman_acc
                # We consider accuracy as the average accuracy over the three types of accuracy
            global_loss /= num_val_chunks
            pose_loss /= num_val_chunks
            object_loss /= num_val_chunks
            human_loss /= num_val_chunks
            pose_acc /= num_val_chunks
            object_acc /= num_val_chunks
            human_acc /= num_val_chunks
            with open(valcsvPath, 'a') as f:
                writer = csv.writer(f)
                acc = (pose_acc + object_acc + human_acc) / 3
                writer.writerow([str(acc), pose_acc, object_acc, human_acc, global_loss, pose_loss, object_loss, human_loss])
            if acc > maxValAcc:
                print("New best acc " + str(acc) + " loss " + str(global_loss))
                model.save(bestModelPath)
                maxValAcc = acc

    if sendmail:
        sendemail(from_addr='pythonscriptsisr@gmail.com',
                  to_addr_list=['pedro_abreu95@hotmail.com', 'joaogamartins@gmail.com'],
                  cc_addr_list=[],
                  subject='Finished training 3-stream fusion',
                  message='Training fusion with following params: ' + str(params),
                  login='pythonscriptsisr@gmail.com',
                  password='1!qwerty')


if __name__ == '__main__':
    main()
