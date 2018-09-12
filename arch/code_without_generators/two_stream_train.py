# Learn merged rgb-stream filters (visualize them)
import tensorflow as tf
import utils
# from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras import backend as K
from two_stream_model import TwoStreamModel
from two_stream_data import get_AVA_set, get_AVA_labels, load_split
import time
import csv
import timeit


def main():
    # root_dir = '../../../AVA2.1/' # root_dir for the files
    root_dir = '../../data/AVA/files/'
    K.clear_session()

    # Load list of action classes and separate them (from utils_stream)
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': (224, 224), 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 200, 'model': 'resnet50', 'email': True,
              'train_chunk_size': 2**10, 'validation_chunk_size': 2**10}
    minValLoss = 9999990.0
    encoding = "rgb"
    crop = True

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['train'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Train_Custom_Corrected.csv", train=True)  # IDs for training
    partition['validation'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Val_Custom_Corrected.csv", train=True)  # IDs for validation

    # Labels
    labels_train = get_AVA_labels(classes, partition, "train", filename=root_dir + "AVA_Train_Custom_Corrected.csv")
    labels_val = get_AVA_labels(classes, partition, "validation", filename=root_dir + "AVA_Val_Custom_Corrected.csv")

    # Create + compile model, load saved weights if they exist
    # rgb_weights = "../models/rgb_gauss_resnet50_1806290918.hdf5"
    # rgb_weights = "../models/rgb_gauss_resnet50_1806290918.hdf5"
    rgb_weights = "../models/rgb_fovea_resnet50_1806301953.hdf5"
    flow_weights = "../models/flowcrop_resnet50_1807180022.hdf5"
    # flow_weights = "../models/flowcrop_resnet50_1807180022.hdf5"

    # Model
    nsmodel = TwoStreamModel(classes['label_id'], rgb_weights, flow_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model

    print("Training set size: " + str(len(partition['train'])))

    # Load splits as chunks (see utils)
    train_splits = utils.make_chunks(original_list=partition['train'], size=len(partition['train']), chunk_size=params['train_chunk_size'])
    val_splits = utils.make_chunks(original_list=partition['validation'], size=len(partition['validation']), chunk_size=params['validation_chunk_size'])
    filter_type = "fovea"
    rgb_dir = "/media/pedro/actv-ssd/" + filter_type + "_"
    flow_dir = "/media/pedro/actv-ssd/flowcrop_"

    # Paths for saving model and plots
    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    bestModelPath = "../models/two_stream_fusion_flowcrop_fovea_" + filter_type + "_" + params['model'] + "_" + time_str + ".hdf5"
    traincsvPath = "../loss_acc_plots/two_stream_fusion_flowcrop_fovea_train_" + filter_type + "_plot_" + params['model'] + "_" + time_str + ".csv"
    valcsvPath = "../loss_acc_plots/two_stream_fusion_flowcrop_fovea_val_" + filter_type + "_plot_" + params['model'] + "_" + time_str + ".csv"

    with tf.device('/gpu:0'):
        for epoch in range(params['nb_epochs']):
            epoch_chunks_count = 0
            for trainIDS in train_splits:
                start_time = timeit.default_timer()
                # -----------------------------------------------------------
                x_val_rgb = x_val_flow = y_val_pose = y_val_object = y_val_human = x_train_rgb = x_train_flow = y_train_pose = y_train_object = y_train_human = None
                x_train_rgb, x_train_flow, y_train_pose, y_train_object, y_train_human = load_split(trainIDS, labels_train, params['dim'], params['n_channels'], 10, rgb_dir, flow_dir, "train", encoding, train=True, crop=crop)

                y_train_pose = to_categorical(y_train_pose, num_classes=utils.POSE_CLASSES)
                y_train_object = utils.to_binary_vector(y_train_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human')
                y_train_human = utils.to_binary_vector(y_train_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human')

                history = model.fit([x_train_rgb, x_train_flow], [y_train_pose, y_train_object, y_train_human], batch_size=params['batch_size'], epochs=1, verbose=0)
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
            print("Validating data: ")
            # global_loss, pose_loss, object_loss, human_loss, pose_acc, object_acc, human_acc
            loss_acc_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            for valIDS in val_splits:
                x_val_rgb = x_val_flow = y_val_pose = y_val_object = y_val_human = x_train_rgb = x_train_flow = y_train_pose = y_train_object = y_train_human = None
                x_val_rgb, x_val_flow, y_val_pose, y_val_object, y_val_human = load_split(valIDS, labels_val, params['dim'], params['n_channels'], 10, rgb_dir, flow_dir, "val", encoding, train=True, crop=crop)

                y_val_pose = to_categorical(y_val_pose, num_classes=utils.POSE_CLASSES)
                y_val_object = utils.to_binary_vector(y_val_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human')
                y_val_human = utils.to_binary_vector(y_val_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human')

                vglobal_loss, vpose_loss, vobject_loss, vhuman_loss, vpose_acc, vobject_acc, vhuman_acc = model.evaluate([x_val_rgb, x_val_flow], [y_val_pose, y_val_object, y_val_human], batch_size=params['batch_size'])
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
                        subject='Finished training and validating fusion',
                        message='Training fusion with following params: ' + str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
