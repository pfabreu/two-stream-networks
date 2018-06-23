# Learn merged rgb-stream filters (visualize them)
import tensorflow as tf
import utils
from keras.utils import to_categorical
from stream_2_model import NStreamModel
from stream_2_data import get_AVA_set, get_AVA_labels, get_AVA_classes, load_split
import time
import csv
import timeit


def main():
    K.clear_session()

    sendmail = True
    train = True

    # Load list of action classes and separate them (from utils_stream)
    classes = get_AVA_classes('AVA2.1/ava_action_list_v2.1.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': (224, 224), 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 150, 'model': 'resnet50', 'email': 'sendmail'}

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['train'] = get_AVA_set(classes=classes, filename="AVA2.1/ava_mini_split_train_big.csv", train=True)  # IDs for training
    partition['validation'] = get_AVA_set(classes=classes, filename="AVA2.1/ava_mini_split_val_big.csv", train=True)  # IDs for validation

    # Labels
    labels_train = get_AVA_labels(classes, partition, "train", filename="AVA2.1/ava_mini_split_train_big.csv")
    labels_val = get_AVA_labels(classes, partition, "validation", filename="AVA2.1/ava_mini_split_val_big.csv")

    # Create + compile model, load saved weights if they exist
    rgb_weights = "rgb_resnet50_1805290059.hdf5"
    flow_weights = "flow_resnet50_1805290120.hdf5"
    nsmodel = NStreamModel(classes['label_id'], rgb_weights, flow_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model

    print("Training set size: " + str(len(partition['train'])))

    # Load first train_size of partition{'train'}
    train_splits = utils.make_chunks(original_list=partition['train'], size=2**15, chunk_size=2**12)
    val_splits = utils.make_chunks(original_list=partition['train'], size=2**15, chunk_size=2**12)
    num_val_chunks = len(val_splits)
    rgb_dir = "/media/pedro/actv-ssd/foveated_train_gc/"
    flow_dir = "test/flow/actv-ssd/flow_train"
    maxValAcc = 0.0
    val_chunks_count = 0
    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    bestModelPath = "fusion_" + params['model'] + "_" + time_str + ".hdf5"
    traincsvPath = "rgb_train_plot_" + params['model'] + "_" + time_str + ".csv"
    valcsvPath = "rgb_val_plot_" + params['model'] + "_" + time_str + ".csv"
    with tf.device('/gpu:0'):
        for epoch in range(params['nb_epochs']):
            if train is True:
                epoch_chunks_count = 0
                for trainIDS in train_splits:
                    start_time = timeit.default_timer()
                    # -----------------------------------------------------------
                    x_val_rgb = x_val_flow = y_val_pose = y_val_object = y_val_human = x_train_rgb = x_train_flow = y_train_pose = y_train_object = y_train_human = None
                    x_train_rgb, x_train_flow, y_train_pose, y_train_object, y_train_human = load_split(trainIDS, labels_train, params['dim'], params['n_channels'], 10, rgb_dir, flow_dir, train)

                    y_train_pose = to_categorical(y_train_pose, num_classes=utils.POSE_CLASSES)
                    y_train_object = utils.to_binary_vector(y_train_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human')
                    y_train_human = utils.to_binary_vector(y_train_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human')

                    history = model.fit([x_train_rgb, x_train_flow], [y_train_pose, y_train_object, y_train_human], batch_size=params['batch_size'], epochs=1, verbose=0)
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
            # global_loss, pose_loss, object_loss, human_loss, pose_acc, object_acc, human_acc
            loss_acc_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            for valIDS in val_splits:
                x_val_rgb = x_val_flow = y_val_pose = y_val_object = y_val_human = x_train_rgb = x_train_flow = y_train_pose = y_train_object = y_train_human = None
                x_val_rgb, x_val_flow, y_val_pose, y_val_object, y_val_human = load_split(valIDS, labels_val, params['dim'], params['n_channels'], 10, rgb_dir, flow_dir, train)
                if train is False:
                    predictions = model.predict([x_val_rgb, x_val_flow], batch_size=32, verbose=1)
                    print("Val chunk " + str(val_chunks_count) + "/" + str(len(val_splits)))
                    utils.pred2classes(valIDS, predictions, output_csv)
                    val_chunks_count += 1
                else:
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
                    # We consider accuracy as the average accuracy over the three types of accuracy
            if train is False:  # We only want to run 1 "epoch" for testing or validation
                break
            loss_acc_list = [x / num_val_chunks for x in loss_acc_list]
            with open(valcsvPath, 'a') as f:
                writer = csv.writer(f)
                acc = (pose_acc + object_acc + human_acc) / 3
                writer.writerow([str(acc), pose_acc, object_acc, human_acc, global_loss, pose_loss, object_loss, human_loss])
            if acc > maxValAcc:
                print("New best acc " + str(acc) + " loss " + str(global_loss))
                model.save(bestModelPath)
                maxValAcc = acc

    if sendmail:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com',
                        to_addr_list=['pedro_abreu95@hotmail.com', 'joaogamartins@gmail.com'],
                        cc_addr_list=[],
                        subject='Finished training/validating fusion',
                        message='Training fusion with following params: ' + str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
