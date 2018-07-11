# Learn merged rgb-stream filters (visualize them)
import tensorflow as tf
import utils
# from tensorflow.python.keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras import backend as K
from stream_3_model import ThreeStreamModel
from stream_3_data import get_AVA_set, get_AVA_labels, load_split
import time
import csv
import timeit


def main():
    root_dir = '../../../AVA2.1/'  # root_dir for the files
    # root_dir = '../../data/AVA/files/'
    K.clear_session()

    # Load list of action classes and separate them (from _stream)
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': (224, 224), 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 150, 'model': 'resnet50', 'email': True,
              'train_chunk_size': 2**11, 'validation_chunk_size': 2**11}
    minValLoss = 9999990.0

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['train'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Train_Custom_Corrected.csv", train=True)  # IDs for training
    partition['validation'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Val_Custom_Corrected.csv", train=True)  # IDs for validation

    # Labels
    labels_train = get_AVA_labels(classes, partition, "train", filename=root_dir + "AVA_Train_Custom_Corrected.csv")
    labels_val = get_AVA_labels(classes, partition, "validation", filename=root_dir + "AVA_Val_Custom_Corrected.csv")

    # Create + compile model, load saved weights if they exist
    rgb_weights = "saved_models/rgb_fovea_resnet50_1806301953.hdf5"
    flow_weights = "saved_models/flow_resnet50_1806281901.hdf5"
    context_weights = "saved_models/bestModelcontext128.hdf5"
    nsmodel = ThreeStreamModel(classes['label_id'], rgb_weights, flow_weights, context_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model
    modelpath = None
    if modelpath is not None:
        print("Loading previous weights")
        model.load_weights(modelpath)

    print("Training set size: " + str(len(partition['train'])))

    # Load splits
    train_splits = utils.make_chunks(original_list=partition['train'], size=len(partition['train']), chunk_size=params['train_chunk_size'])
    val_splits = utils.make_chunks(original_list=partition['validation'], size=len(partition['validation']), chunk_size=params['validation_chunk_size'])
    rgb_dir = "../../../foveated_"
    flow_dir = "../../../flow_"

    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    bestModelPath = "three_stream_fusion_elfov" + params['model'] + "_" + time_str + ".hdf5"
    traincsvPath = "three_stream_train_elfov_plot_" + params['model'] + "_" + time_str + ".csv"
    valcsvPath = "three_stream_val_elfov_plot_" + params['model'] + "_" + time_str + ".csv"

    print("Building context dictionaries from context files (these should be generated)...")
    Xfilename = root_dir + "context_files/XContext_train_pastfuture.csv"
    train_context_rows = {}
    with open(Xfilename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            rkey = row[0] + "_" + row[1].lstrip("0") + \
                "@" + str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
            train_context_rows[rkey] = row[6]

    Xfilename = root_dir + "context_files/XContext_val_pastfuture.csv"
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
                x_train_rgb, x_train_flow, x_train_context, y_train_pose, y_train_object, y_train_human = load_split(trainIDS, labels_train, params['dim'], params['n_channels'], "train", 10, train_context_rows, rgb_dir, flow_dir)

                y_train_pose = to_categorical(y_train_pose, num_classes=utils.POSE_CLASSES)
                y_train_object = utils.to_binary_vector(y_train_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human')
                y_train_human = utils.to_binary_vector(y_train_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human')

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
            loss_acc_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for valIDS in val_splits:
                x_val_rgb = x_val_flow = x_val_context = y_val_pose = y_val_object = y_val_human = x_train_rgb = x_train_flow = y_train_pose = y_train_object = y_train_human = None
                x_val_rgb, x_val_flow, x_val_context, y_val_pose, y_val_object, y_val_human = load_split(valIDS, labels_val, params['dim'], params['n_channels'], "val", 10, val_context_rows, rgb_dir, flow_dir)

                y_val_pose = to_categorical(y_val_pose, num_classes=utils.POSE_CLASSES)
                y_val_object = utils.to_binary_vector(y_val_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human')
                y_val_human = utils.to_binary_vector(y_val_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human')

                vglobal_loss, vpose_loss, vobject_loss, vhuman_loss, vpose_acc, vobject_acc, vhuman_acc = model.evaluate([x_val_rgb, x_val_flow, x_val_context], [y_val_pose, y_val_object, y_val_human], batch_size=params['batch_size'])
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
                        to_addr_list=['pedro_abreu95@hotmail.com', 'joaogamartins@gmail.com'],
                        cc_addr_list=[],
                        subject='Finished training 3-stream fusion',
                        message='Training fusion with following params: ' + str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
