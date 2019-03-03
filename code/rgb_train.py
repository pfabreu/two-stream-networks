import tensorflow as tf
from keras.utils import to_categorical
from keras import backend as K
import csv
import numpy as np

import utils
from rgb_model import rgb_create_model, compile_model
from rgb_data import load_split, get_set, get_labels, RGBDataGenerator


def main():
    root_dir_info = '../../data/AVA/files/'
    root_dir_data = "/media/pedro/actv-ssd/miniAVA/"
    dataset = "AVA"
    # Parameters for training
    params = {'dim': (224, 224), 'batch_size': 32,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 200, 'base_arch': 'resnet50',
              'freeze': False, 'fusion': 'conv', 'train_chunk_size': 2**12,
              'loss': "categorical", 'email': True} # categorical, sigmoids, generalized
              
    filter_type = "gauss"
    bestModelPath = "../models/rgb_sigmoid_" + filter_type + "_" + params['model'] + "_" + time_str + ".hdf5"
    traincsvPath = "../loss_plots/rgb_sigmoid_train_" + filter_type + "_plot_" + params['model'] + "_" + time_str + ".csv"
    valcsvPath = "../loss_plots/rgb_sigmoid_val_" + filter_type + "_plot_" + params['model'] + "_" + time_str + ".csv"

    ############################################################################

    K.clear_session() # Erase previous models from GPU memory

    # Load list of action classes and separate them
    classes = utils.get_classes(dataset=dataset, classlist=root_dir + 'ava_action_list_custom.csv')
    minValLoss = 9999990.0

    # Get ID's from the actual dataset
    partition = {}
    partition['train'] = get_set(dataset=dataset, classes=classes, filename=root_dir + "AVA_Train_Custom_Corrected.csv")  # IDs for training
    partition['validation'] = get_set(dataset=dataset, classes=classes, filename=root_dir + "AVA_Val_Custom_Corrected.csv")  # IDs for validation

    # Labels
    labels_train = get_labels(dataset, partition, "train", filename=root_dir + "AVA_Train_Custom_Corrected.csv", loss=loss)
    labels_val = get_labels(dataset, partition, "validation", filename=root_dir + "AVA_Val_Custom_Corrected.csv", loss=loss)

    # Create + compile model, load saved weights if they exist
    model, keras_layer_names = rgb_create_model(classes=classes['label_id'], loss=loss, model_name=params['model'], freeze=params['freeze_all'], fusion=params['fusion'])
    model = compile_model(model, loss=loss)
    saved_weights = "../models/ucf_keras/keras-ucf101-rgb-resnet50-sigmoids.hdf5"
    if saved_weights is not None:
        model.load_weights(saved_weights)

    #print("Training set size: " + str(len(partition['train'])))

    # Data generators
    training_generator = RGBDataGenerator(partition['train'], labels, **params)
    validation_generator = RGBDataGenerator(partition['validation'], labels, **params)

    for epoch in range(params['nb_epochs']):
        epoch_chunks_count = 0
        for trainIDS in train_splits:
            # Load and train
            start_time = timeit.default_timer()
            # -----------------------------------------------------------
            x_val = y_val_pose = y_val_object = y_val_human = x_train = y_train_pose = y_train_object = y_train_human = None
            x_train, y_train_pose, y_train_object, y_train_human = load_split(trainIDS, labels_train, params[
                'dim'], params['n_channels'], "train", filter_type, loss=loss)

            y_t = []

            y_t.append(to_categorical(y_train_pose, num_classes=utils.POSE_CLASSES))
            y_t.append(utils.to_binary_vector(y_train_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
            y_t.append(utils.to_binary_vector(y_train_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))
            if loss is False:
                # print(y_t[0].shape)
                # print(y_t[1].shape)
                # print(y_t[2].shape)
                y_t = np.concatenate((y_t[0], y_t[1], y_t[2]), axis=1)
                # print(y_t.shape)
            history = model.fit(x_train, y_t, batch_size=params['batch_size'], epochs=1, verbose=0)
            utils.learning_rate_schedule(model, epoch, params['nb_epochs'])

            # TODO Repeat samples of unrepresented classes?

            # ------------------------------------------------------------
            elapsed = timeit.default_timer() - start_time
            if loss is True:
                print("Epoch " + str(epoch) + " chunk " + str(epoch_chunks_count) + " (" + str(elapsed) + ") acc[pose,obj,human] = [" + str(history.history['pred_pose_categorical_accuracy']) + "," +
                      str(history.history['pred_obj_human_categorical_accuracy']) + "," + str(history.history['pred_human_human_categorical_accuracy']) + "] loss: " + str(history.history['loss']))
            else:
                print("Epoch " + str(epoch) + " chunk " + str(epoch_chunks_count) + " (" + str(elapsed) + ") acc = [" + str(history.history['categorical_accuracy']) + "," + "] loss: " + str(history.history['loss']))
            with open(traincsvPath, 'a') as f:
                writer = csv.writer(f)
                if loss is True:
                    avg_acc = (history.history['pred_pose_categorical_accuracy'][0] + history.history['pred_obj_human_categorical_accuracy'][0] + history.history['pred_human_human_categorical_accuracy'][0]) / 3
                    writer.writerow([str(avg_acc), history.history['pred_pose_categorical_accuracy'], history.history['pred_obj_human_categorical_accuracy'], history.history['pred_human_human_categorical_accuracy'], history.history['loss']])
                else:
                    avg_acc = history.history['categorical_accuracy'][0]
                    writer.writerow([str(avg_acc), history.history['loss']])
            epoch_chunks_count += 1
        # Load val_data
        print("Validating data: ")
        # global_loss, pose_loss, object_loss, human_loss, pose_acc, object_acc, human_acc
        loss_acc_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for valIDS in val_splits:
            x_val = y_val_pose = y_val_object = y_val_human = x_train = y_train_pose = y_train_object = y_train_human = None
            x_val, y_val_pose, y_val_object, y_val_human = load_split(valIDS, labels_val, params['dim'], params['n_channels'], "val", filter_type, loss=loss)

            y_v = []
            y_v.append(to_categorical(y_val_pose, num_classes=utils.POSE_CLASSES))
            y_v.append(utils.to_binary_vector(y_val_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
            y_v.append(utils.to_binary_vector(y_val_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))
            if loss is False:
                y_v = np.concatenate((y_v[0], y_v[1], y_v[2]), axis=1)

            if loss is True:
                vglobal_loss, vpose_loss, vobject_loss, vhuman_loss, vpose_acc, vobject_acc, vhuman_acc = model.evaluate(x_val, y_v, batch_size=params['batch_size'])
                loss_acc_list[0] += vglobal_loss
                loss_acc_list[1] += vpose_loss
                loss_acc_list[2] += vobject_loss
                loss_acc_list[3] += vhuman_loss
                loss_acc_list[4] += vpose_acc
                loss_acc_list[5] += vobject_acc
                loss_acc_list[6] += vhuman_acc
            else:
                vglobal_loss, v_acc = model.evaluate(x_val, y_v, batch_size=params['batch_size'])
                loss_acc_list[0] += vglobal_loss
                loss_acc_list[1] += v_acc

        # Average over all validation chunks
        loss_acc_list = [x / len(val_splits) for x in loss_acc_list]
        with open(valcsvPath, 'a') as f:
            writer = csv.writer(f)
            if loss is True:
                acc = (loss_acc_list[4] + loss_acc_list[5] + loss_acc_list[6]) / 3
                writer.writerow([str(acc), loss_acc_list[4], loss_acc_list[5], loss_acc_list[6], loss_acc_list[0], loss_acc_list[1], loss_acc_list[2], loss_acc_list[3]])
            else:
                acc = loss_acc_list[1]
                writer.writerow([str(acc), loss_acc_list[0]])
        if loss_acc_list[0] < minValLoss:
            print("New best loss " + str(loss_acc_list[0]))
            model.save(bestModelPath)
            minValLoss = loss_acc_list[0]

    if params['email']:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com', to_addr_list=['pedro_abreu95@hotmail.com'],
                        subject='Finished training RGB', message='Training RGB with following params: ' + str(params),
                        login='pythonscriptsisr@gmail.com', password='1!qwerty')

if __name__ == '__main__':
    main()
