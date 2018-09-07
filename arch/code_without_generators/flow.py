import tensorflow as tf
# from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras import backend as K
import csv
import time
import timeit
import os

import utils
from flow_model import flow_create_model, compile_model
from flow_data import get_AVA_set, get_AVA_labels, load_split


def main():
    # root_dir for the files
    root_dir = '../../data/AVA/files/'

    # Erase previous models from GPU memory
    K.clear_session()

    # Load list of action classes and separate them (from utils_stream)
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': (224, 224), 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 10,
              'nb_epochs': 157, 'model': "inceptionv3", 'email': True,
              'freeze_all': True, 'conv_fusion': False, 'train_chunk_size': 2**10,
              'validation_chunk_size': 2**10}
    minValLoss = 9999990.0
    soft_sigmoid = True
    warp = False  # TODO Use warped (camera motion corrected) flow or not
    crop = False  # TODO Use crop flow or not
    encoding = "rgb"  # TODO Are flows stored as rgb or as 2 grayscales

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['train'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Train_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)  # IDs for training
    partition['validation'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Val_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)  # IDs for validation

    # Labels
    labels_train = get_AVA_labels(classes, partition, "train", filename=root_dir + "AVA_Train_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)
    labels_val = get_AVA_labels(classes, partition, "validation", filename=root_dir + "AVA_Val_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)

    # Create + compile model, load saved weights if they exist
    # saved_weights = "saved_models/RGB_Stream_Softmax_inceptionv3.hdf5"
    saved_weights = "../models/flow_kineticsinit_inceptionv3_1808282350.hdf5"
    ucf_weights = "../models/ucf_keras/keras-ucf101-TVL1flow-" + params['model'] + "-newsplit.hdf5"  # Outdated model
    # ucf_weights = None

    # ucf_weights = None
    model, keras_layer_names = flow_create_model(classes=classes['label_id'], model_name=params['model'], soft_sigmoid=soft_sigmoid, image_shape=(224, 224), opt_flow_len=10, freeze_all=params['freeze_all'], conv_fusion=params['conv_fusion'])
    model = compile_model(model, soft_sigmoid=soft_sigmoid)

    # TODO Experiment: 1. no initialization, 2. ucf initialization 3. kinetics initialization
    initialization = True  # Set to True to use initialization
    kinetics_weights = None
    ucf_weights = ""

    if saved_weights is not None:
        model.load_weights(saved_weights)
    else:
        if initialization is True:
            if kinetics_weights is None:
                if params['model'] == "inceptionv3":
                    print("Loading kinetics weights: ")
                    keras_weights = ["../models/kinetics_keras/tsn_flow_params_names.pkl", "../models/kinetics_keras/tsn_flow_params.pkl"]
                    utils.convert_inceptionv3(model, keras_weights, keras_layer_names)
                    model.save("../models/kinetics_keras/keras-kinetics-flow-inceptionv3.hdf5")

            if ucf_weights is None:
                print("Loading UCF weights: ")
                if params['model'] == "resnet50":
                    # TODO Better initialization, average UCF models overt he 3 splits provided
                    ucf_weights = utils.loadmat("../models/ucf_matconvnet/ucf101-TVL1flow-resnet-50-split1.mat")
                    utils.convert_resnet(model, ucf_weights)
                    model.save("../models/ucf_keras/keras-ucf101-TVL1flow-resnet50-newsplit.hdf5")
            else:
                if ucf_weights != "":
                    model.load_weights(ucf_weights)

    # Try to train on more than 1 GPU if possible
    # try:
    #    print("Trying MULTI-GPU")
    #    model = multi_gpu_model(model)
    print("Training set size: " + str(len(partition['train'])))

    # Load first train_size of partition{'train'}
    train_splits = utils.make_chunks(original_list=partition['train'], size=len(partition['train']), chunk_size=params['train_chunk_size'])
    val_splits = utils.make_chunks(original_list=partition['validation'], size=len(partition['validation']), chunk_size=params['validation_chunk_size'])

    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    if crop is True:
        bestModelPath = "../models/flowcrop_" + params['model'] + "_" + time_str + ".hdf5"
        traincsvPath = "../loss_acc_plots/flowcrop_customcsv_train_plot_" + params['model'] + "_" + time_str + ".csv"
        valcsvPath = "../loss_acc_plots/flowcrop_customcsv_val_plot_" + params['model'] + "_" + time_str + ".csv"
    else:
        if warp is True:
            bestModelPath = "../models/flow_warp_" + params['model'] + "_" + time_str + ".hdf5"
            traincsvPath = "../loss_acc_plots/flow_warp_customcsv_train_plot_" + params['model'] + "_" + time_str + ".csv"
            valcsvPath = "../loss_acc_plots/flow_warp_customcsv_val_plot_" + params['model'] + "_" + time_str + ".csv"
        else:
            bestModelPath = "../models/flow_kineticsinit_" + params['model'] + "_" + time_str + ".hdf5"
            traincsvPath = "../loss_acc_plots/flow_kineticsinit_customcsv_train_plot_" + params['model'] + "_" + time_str + ".csv"
            valcsvPath = "../loss_acc_plots/flow_kineticsinit_customcsv_val_plot_" + params['model'] + "_" + time_str + ".csv"
    first_epoch = True

    with tf.device('/gpu:0'):  # TODO Multi GPU
        for epoch in range(params['nb_epochs']):
            epoch_chunks_count = 0
            if epoch > 0:
                first_epoch = False
            for trainIDS in train_splits:
                start_time = timeit.default_timer()
                # -----------------------------------------------------------
                x_val = y_val_pose = y_val_object = y_val_human = x_train = y_train_pose = y_train_object = y_train_human = None
                x_train, y_train_pose, y_train_object, y_train_human = load_split(trainIDS, labels_train, params['dim'], params['n_channels'], "train", 5, first_epoch, encoding=encoding, soft_sigmoid=soft_sigmoid, crop=True)

                y_t = []
                y_t.append(to_categorical(y_train_pose, num_classes=utils.POSE_CLASSES))
                y_t.append(utils.to_binary_vector(y_train_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
                y_t.append(utils.to_binary_vector(y_train_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))
                history = model.fit(x_train, y_t, batch_size=params['batch_size'], epochs=1, verbose=0)
                utils.learning_rate_schedule(model, epoch, params['nb_epochs'])
                # ------------------------------------------------------------
                elapsed = timeit.default_timer() - start_time

                print("Epoch " + str(epoch) + " chunk " + str(epoch_chunks_count) + " (" + str(elapsed) + ") acc[pose,obj,human] = [" + str(history.history['pred_pose_categorical_accuracy']) + "," +
                      str(history.history['pred_obj_human_categorical_accuracy']) + "," + str(history.history['pred_human_human_categorical_accuracy']) + "] loss: " + str(history.history['loss']))
                with open(traincsvPath, 'a') as f:
                    writer = csv.writer(f)
                    avg_acc = (history.history['pred_pose_categorical_accuracy'][0] + history.history['pred_obj_human_categorical_accuracy'][0] + history.history['pred_human_human_categorical_accuracy'][0]) / 3
                    writer.writerow([str(avg_acc), history.history['pred_pose_categorical_accuracy'], history.history['pred_obj_human_categorical_accuracy'], history.history['pred_human_human_categorical_accuracy'], history.history['loss']])

                epoch_chunks_count += 1
            # Load val_data
            print("Validating data: ")
            loss_acc_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for valIDS in val_splits:
                x_val = y_val_pose = y_val_object = y_val_human = x_train = y_train_pose = y_train_object = y_train_human = None
                x_val, y_val_pose, y_val_object, y_val_human = load_split(valIDS, labels_val, params['dim'], params['n_channels'], "val", 5, first_epoch, encoding=encoding, soft_sigmoid=soft_sigmoid, crop=True)

                y_v = []
                y_v.append(to_categorical(y_val_pose, num_classes=utils.POSE_CLASSES))
                y_v.append(utils.to_binary_vector(y_val_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
                y_v.append(utils.to_binary_vector(y_val_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))
                vglobal_loss, vpose_loss, vobject_loss, vhuman_loss, vpose_acc, vobject_acc, vhuman_acc = model.evaluate(x_val, y_v, batch_size=params['batch_size'])
                loss_acc_list[0] += vglobal_loss
                loss_acc_list[1] += vpose_loss
                loss_acc_list[2] += vobject_loss
                loss_acc_list[3] += vhuman_loss
                loss_acc_list[4] += vpose_acc
                loss_acc_list[5] += vobject_acc
                loss_acc_list[6] += vhuman_acc
            loss_acc_list = [x / len(val_splits) for x in loss_acc_list]
            with open(valcsvPath, 'a') as f:
                writer = csv.writer(f)
                acc = (loss_acc_list[4] + loss_acc_list[5] + loss_acc_list[6]) / 3
                writer.writerow([str(acc), loss_acc_list[4], loss_acc_list[5], loss_acc_list[6], loss_acc_list[0], loss_acc_list[1], loss_acc_list[2], loss_acc_list[3]])
            if loss_acc_list[0] < minValLoss:
                print("New best loss " + str(loss_acc_list[0]))
                model.save(bestModelPath)
                minValLoss = loss_acc_list[0]

    if params['email']:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com',
                        to_addr_list=['pedro_abreu95@hotmail.com'],
                        subject='Finished training OF-stream ',
                        message='Training OF with following params: ' + str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
