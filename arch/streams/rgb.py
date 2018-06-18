from tensorflow.python.keras.utils import to_categorical, multi_gpu_model
from tensorflow.python.keras import backend as K
import csv
import time
import timeit

import utils
from rgb_model import rgb_create_model, compile_model
from rgb_data import load_split, get_AVA_set, get_AVA_labels


def main():
    root_dir = '../../data/AVA/files/'
    # Erase previous models from GPU memory
    K.clear_session()

    sendmail = True
    soft_sigmoid = True

    # Load list of action classes and separate them
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training
    params = {'dim': (224, 224), 'batch_size': 32,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 200, 'model': 'resnet50'}

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['train'] = get_AVA_set(classes=classes, filename=root_dir +"AVA_Train_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)  # IDs for training

    partition['validation'] = get_AVA_set(classes=classes, filename=root_dir +"AVA_Val_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)  # IDs for validation
    # Labels
    labels_train = get_AVA_labels(
        classes, partition, "train", filename=root_dir +"ava_mini_split_train_big.csv", soft_sigmoid=soft_sigmoid)
    labels_val = get_AVA_labels(classes, partition, "validation",
                                filename=root_dir +"AVA_Val_Custom_Corrected.csv", soft_sigmoid=soft_sigmoid)

    # Create + compile model, load saved weights if they exist
    saved_weights = None
    model = rgb_create_model(
        classes=classes['label_id'], soft_sigmoid=soft_sigmoid, model_name=params['model'], freeze_all=False)
    model = compile_model(model, soft_sigmoid=soft_sigmoid)
    if saved_weights is not None:
        model.load_weights(saved_weights)
    # Try to train on more than 1 GPU if possible
    # try:
    #    print("Trying MULTI-GPU")
    #    model = multi_gpu_model(model)

    print("Training set size: " + str(len(partition['train'])))

    # Load first train_size of partition{'train'}

    train_splits = utils.make_chunks(
        original_list=partition['train'], size=2**15, chunk_size=2**12)
    val_splits = utils.make_chunks(
        original_list=partition['validation'], size=2**15, chunk_size=2**12)
    num_val_chunks = len(val_splits)

    minValLoss = 0.0
    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    bestModelPath = "../models/rgb_" + params['model'] + "_" + time_str + ".hdf5"
    traincsvPath = "../plots/rgb_train_plot_" + \
        params['model'] + "_" + time_str + ".csv"
    valcsvPath = "../plots/rgb_val_plot_" + \
        params['model'] + "_" + time_str + ".csv"
    # with tf.device('/gpu:0'):
    for epoch in range(params['nb_epochs']):
        epoch_chunks_count = 0
        for trainIDS in train_splits:
            # Load and train
            if soft_sigmoid is False:
                # can do this because None is a singleton, yay
                x_val = y_val = x_train = y_train = None
                x_train, y_train = load_split(
                    trainIDS, labels_train, params['dim'], params['n_channels'], "train", soft_sigmoid=soft_sigmoid)
                y_train = to_categorical(
                    y_train, num_classes=params['n_classes'])
                history = model.fit(
                    x_train, y_train, batch_size=params['batch_size'], epochs=1, verbose=0)
                print("Epoch " + str(epoch) + " chunk " + str(epoch_chunks_count) + " acc: " +
                      str(history.history['acc']) + " loss: " + str(history.history['loss']))
            else:
                start_time = timeit.default_timer()
                # -----------------------------------------------------------
                x_val = y_val_pose = y_val_object = y_val_human = x_train = y_train_pose = y_train_object = y_train_human = None
                x_train, y_train_pose, y_train_object, y_train_human = load_split(
                    trainIDS, labels_train, params['dim'], params['n_channels'], "train", soft_sigmoid=soft_sigmoid)

                y_t = []
                y_t.append(to_categorical(
                    y_train_pose, num_classes=utils.POSE_CLASSES))
                y_t.append(utils.to_binary_vector(
                    y_train_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
                y_t.append(utils.to_binary_vector(
                    y_train_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))

                history = model.fit(
                    x_train, y_t, batch_size=params['batch_size'], epochs=1, verbose=0)
                utils.learning_rate_schedule(model, epoch, params['nb_epochs'])
                elapsed = timeit.default_timer() - start_time
                # ------------------------------------------------------------
                # Rough estimate of time
                eta_time = (params['nb_epochs'] - epoch) * elapsed * len(train_splits)
                print("Epoch " + str(epoch) + " chunk " + str(epoch_chunks_count) + " (" + str(elapsed) + ", ETA: )" + str(eta_time) + " acc[pose,obj,human] = [" + str(history.history['pred_pose_categorical_accuracy']) + "," +
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
            if soft_sigmoid is False:
                # can do this because None is a singleton, yay
                x_val = y_val = x_train = y_train = None
                # Test on a random chunk
                x_val, y_val = load_split(
                    valIDS, labels_val, params['dim'], params['n_channels'], "validation", soft_sigmoid=soft_sigmoid)
                y_val = to_categorical(y_val, num_classes=params['n_classes'])
                global_loss, acc = model.evaluate(
                    x_val, y_val, batch_size=params['batch_size'])
            else:
                x_val = y_val_pose = y_val_object = y_val_human = x_train = y_train_pose = y_train_object = y_train_human = None
                # x_val, y_val_pose, y_val_object, y_val_human = load_split(random.choice(val_splits), labels_val, params['dim'], params['n_channels'], "val", soft_sigmoid=soft_sigmoid)
                x_val, y_val_pose, y_val_object, y_val_human = load_split(
                    valIDS, labels_val, params['dim'], params['n_channels'], "val", soft_sigmoid=soft_sigmoid)
                y_v = []
                y_v.append(to_categorical(
                    y_val_pose, num_classes=utils.POSE_CLASSES))
                y_v.append(utils.to_binary_vector(
                    y_val_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
                y_v.append(utils.to_binary_vector(
                    y_val_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))

                vglobal_loss, vpose_loss, vobject_loss, vhuman_loss, vpose_acc, vobject_acc, vhuman_acc = model.evaluate(x_val, y_v, batch_size=params['batch_size'])
                loss_acc_list[0] += vglobal_loss
                loss_acc_list[1] += vpose_loss
                loss_acc_list[2] += vobject_loss
                loss_acc_list[3] += vhuman_loss
                loss_acc_list[4] += vpose_acc
                loss_acc_list[5] += vobject_acc
                loss_acc_list[6] += vhuman_acc
        # Average over all validation chunks
        loss_acc_list = [x / num_val_chunks for x in loss_acc_list]
        with open(valcsvPath, 'a') as f:
            writer = csv.writer(f)
            # We consider accuracy as the average accuracy over the three types of accuracy
            acc = (loss_acc_list[4] + loss_acc_list[5] + loss_acc_list[6]) / 3
            writer.writerow([str(acc), loss_acc_list[4], loss_acc_list[5], loss_acc_list[6],loss_acc_list[0], loss_acc_list[1], loss_acc_list[2], loss_acc_list[3]])
        if loss_acc_list[0] < minValLoss:
            print("New best loss " + str(loss_acc_list[0]))
            model.save(bestModelPath)
            minValLoss = loss_acc_list[0]

    if sendmail:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com',
                        to_addr_list=['pedro_abreu95@hotmail.com',
                                      'joaogamartins@gmail.com'],
                        cc_addr_list=[],
                        subject='Finished training RGB-stream (para fazer fusion convolucional) desculpa o spam',
                        message='Training RGB with following params: ' +
                        str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
