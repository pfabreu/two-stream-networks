import tensorflow as tf
# from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras import backend as K
import csv
import time
import timeit

import utils
from rgb_model import rgb_create_model, compile_model
from rgb_data_aug import load_split, get_AVA_set, get_AVA_labels


def oversampling(classes, root_dir, file):
    types = np.zeros(len(classes['label_id']))

    avg_samples = 0
    with open(root_dir + file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            tp = int(row[6]) - 1
            types[tp] += 1
            avg_samples += 1
    avg_samples /= len(classes['label_id'])

    print(types)
    print(avg_samples)
    fds = 0
    classes_to_rep = []
    reps = []
    for i in range(len(types)):
        if types[i] < avg_samples and types[i] != 0:
            # print("Class: " + str(i + 1))
            # print("Samples: " + str(types[i]))
            # print("Reps: " + str(math.ceil(avg_samples / types[i])))
            reps.append(int(math.ceil(avg_samples / types[i])) - 1)
            classes_to_rep.append(i + 1)
            fds += 1
    print(classes_to_rep)
    print(reps)

    g = sns.barplot(x=[str(i) for i in classes_to_rep], y=reps)
    plt.xticks(rotation=-90)
    plt.title(file + " reps, with avg " + str(avg_samples))
    plt.grid(True)
    plt.show()
    # TODO Histogram to show how many reps per class
    samples = []
    with open(root_dir + file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            tp = int(row[6])
            cidx = 0
            for c in classes_to_rep:
                if tp == c:
                    for r in range(reps[cidx]):
                        row[0] = "#" + row[0]
                        samples.append(row)
                cidx += 1

                # Find all labels in AVA_Train_Custom_Corrected that correspond to each of these classes
    return samples, classes_to_rep


def undersampling(classes, root_dir, file):

    avg_samples = 0
    with open(root_dir + file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            tp = int(row[6]) - 1
            types[tp] += 1
            avg_samples += 1
    avg_samples /= len(classes['label_id'])

    print(types)
    print(avg_samples)

    classes_to_remove = []
    removes = []
    for i in range(len(types)):
        if types[i] > avg_samples:
            reps.append(int(math.ceil(avg_samples / types[i])) - 1)
            classes_to_rep.append(i + 1)
            fds += 1
    print(classes_to_rep)
    print(reps)


def main():
    # root_dir = '../../../AVA2.1/' # root_dir for the files
    root_dir = '../../data/AVA/files/'

    # Erase previous models from GPU memory
    K.clear_session()

    # Load list of action classes and separate them
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training
    params = {'dim': (224, 224), 'batch_size': 32,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 200, 'model': 'resnet50', 'email': True,
              'freeze_all': True, 'conv_fusion': False, 'train_chunk_size': 2**12,
              'validation_chunk_size': 2**12}
    soft_sigmoid = True
    minValLoss = 9999990.0

    # TODO Oversampling
    aug_train, aug_train_classes = oversampling(classes, root_dir, "AVA_Train_Custom_Corrected.csv")
    aug_val, aug_val_classes = oversampling(classes, root_dir, "AVA_Val_Custom_Corrected.csv")

    #  TODO Undersampling

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['train'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Train_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)  # IDs for training
    partition['validation'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Val_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)  # IDs for validation

    # Labels
    labels_train = get_AVA_labels(classes, partition, "train", filename=root_dir + "AVA_Train_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)
    labels_val = get_AVA_labels(classes, partition, "validation", filename=root_dir + "AVA_Val_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)

    partition['train'] = partition['train'] + aug_train
    partition['validation'] = partition['validation'] + aug_val

    # Create + compile model, load saved weights if they exist
    saved_weights = None
    # saved_weights = "../models/rgbextra_gauss_resnet50_1807250030.hdf5"
    model, keras_layer_names = rgb_create_model(classes=classes['label_id'], soft_sigmoid=soft_sigmoid, model_name=params['model'], freeze_all=params['freeze_all'], conv_fusion=params['conv_fusion'])
    model = compile_model(model, soft_sigmoid=soft_sigmoid)

    # TODO Experiment: 1. no initialization, 2. ucf initialization 3. kinetics initialization
    initialization = True  # Set to True to use initialization
    kinetics_weights = None
    ucf_weights = "a"

    if saved_weights is not None:
        model.load_weights(saved_weights)
    else:
        if initialization is True:
            if ucf_weights is None:
                print("Loading MConvNet weights: ")
                if params['model'] == "resnet50":
                    ucf_weights = utils.loadmat("../models/ucf_matconvnet/ucf101-img-resnet-50-split1.mat")
                    utils.convert_resnet(model, ucf_weights)
                    model.save("../models/ucf_keras/keras-ucf101-rgb-resnet50-newsplit.hdf5")
            if kinetics_weights is None:
                if params['model'] == "inceptionv3":
                    print("Loading Keras weights: ")
                    keras_weights = ["../models/kinetics_keras/tsn_rgb_params_names.pkl", "../models/kinetics_keras/tsn_rgb_params.pkl"]
                    utils.convert_inceptionv3(model, keras_weights, keras_layer_names)
                    model.save("../models/kinetics_keras/keras-kinetics-rgb-inceptionv3.hdf5")
    # Try to train on more than 1 GPU if    possible
    # try:
    #    print("Trying MULTI-GPU")
    #    model = multi_gpu_model(model)

    print("Training set size: " + str(len(partition['train'])))

    # Make spltis
    train_splits = utils.make_chunks(original_list=partition['train'], size=len(partition['train']), chunk_size=params['train_chunk_size'])
    val_splits = utils.make_chunks(original_list=partition['validation'], size=len(partition['validation']), chunk_size=params['validation_chunk_size'])

    time_str = time.strftime("%y%m%d%H%M", time.localtime())

    # TODO Don't forget to change your names :)
    filter_type = "gauss"
    bestModelPath = "../models/rgb_augsampling_" + filter_type + "_" + params['model'] + "_" + time_str + ".hdf5"
    traincsvPath = "../plots/rgb_augsampling_train_" + filter_type + "_plot_" + params['model'] + "_" + time_str + ".csv"
    valcsvPath = "../plots/rgb_augsampling_val_" + filter_type + "_plot_" + params['model'] + "_" + time_str + ".csv"

    with tf.device('/gpu:0'):  # NOTE Not using multi gpu
        for epoch in range(params['nb_epochs']):
            epoch_chunks_count = 0
            for trainIDS in train_splits:
                # Load and train
                start_time = timeit.default_timer()
                # -----------------------------------------------------------
                x_val = y_val_pose = y_val_object = y_val_human = x_train = y_train_pose = y_train_object = y_train_human = None
                x_train, y_train_pose, y_train_object, y_train_human = load_split(trainIDS, labels_train, params[
                    'dim'], params['n_channels'], "train", filter_type, soft_sigmoid=soft_sigmoid)

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
                    avg_acc = (history.history['pred_pose_categorical_accuracy'][0] + history.history['pred_obj_human_categorical_accuracy']
                               [0] + history.history['pred_human_human_categorical_accuracy'][0]) / 3
                    writer.writerow([str(avg_acc), history.history['pred_pose_categorical_accuracy'], history.history['pred_obj_human_categorical_accuracy'],
                                     history.history['pred_human_human_categorical_accuracy'], history.history['loss']])
                epoch_chunks_count += 1
            # Load val_data
            print("Validating data: ")
            # global_loss, pose_loss, object_loss, human_loss, pose_acc, object_acc, human_acc
            loss_acc_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for valIDS in val_splits:
                x_val = y_val_pose = y_val_object = y_val_human = x_train = y_train_pose = y_train_object = y_train_human = None
                x_val, y_val_pose, y_val_object, y_val_human = load_split(valIDS, labels_val, params[
                    'dim'], params['n_channels'], "val", filter_type, soft_sigmoid=soft_sigmoid)
                y_v = []
                y_v.append(to_categorical(
                    y_val_pose, num_classes=utils.POSE_CLASSES))
                y_v.append(utils.to_binary_vector(
                    y_val_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
                y_v.append(utils.to_binary_vector(
                    y_val_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))

                vglobal_loss, vpose_loss, vobject_loss, vhuman_loss, vpose_acc, vobject_acc, vhuman_acc = model.evaluate(
                    x_val, y_v, batch_size=params['batch_size'])
                loss_acc_list[0] += vglobal_loss
                loss_acc_list[1] += vpose_loss
                loss_acc_list[2] += vobject_loss
                loss_acc_list[3] += vhuman_loss
                loss_acc_list[4] += vpose_acc
                loss_acc_list[5] += vobject_acc

                loss_acc_list[6] += vhuman_acc
            # Average over all validation chunks
            loss_acc_list = [x / len(val_splits) for x in loss_acc_list]
            with open(valcsvPath, 'a') as f:
                writer = csv.writer(f)
                # We consider accuracy as the average accuracy over the three
                # types of accuracy
                acc = (loss_acc_list[4] +
                       loss_acc_list[5] + loss_acc_list[6]) / 3
                writer.writerow([str(acc), loss_acc_list[4], loss_acc_list[5], loss_acc_list[6],
                                 loss_acc_list[0], loss_acc_list[1], loss_acc_list[2], loss_acc_list[3]])
            if loss_acc_list[0] < minValLoss:
                print("New best loss " + str(loss_acc_list[0]))
                model.save(bestModelPath)
                minValLoss = loss_acc_list[0]

    if params['email']:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com',
                        to_addr_list=['pedro_abreu95@hotmail.com'],
                        subject='Finished training RGB-stream with over and undersampling',
                        message='Training RGB with following params: ' +
                        str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
