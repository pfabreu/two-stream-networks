import tensorflow as tf
import utils
from stream_2_model import TwoStreamModel
from stream_2_data import get_AVA_set, get_AVA_classes, load_split
import time


def main():

    # Load list of action classes and separate them (from utils_stream)
    classes = get_AVA_classes('AVA2.1/ava_action_list_v2.1.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': (224, 224), 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 150, 'model': 'resnet50', 'sendmail': True, 'gen_type': 'val'}

    # Get validation set from directory
    partition = {}
    partition['test'] = get_AVA_set(classes=classes, filename="AVA2.1/ava_mini_split_" + params['gen_type'] + "_big.csv", train=False)
    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    output_csv = "output_" + params['gen_type'] + "_" + time_str + ".csv"

    # Create + compile model, load saved weights if they exist
    rgb_weights = "models/rgb_resnet50_1805290059.hdf5"
    flow_weights = "models/flow_resnet50_1805290120.hdf5"
    nsmodel = TwoStreamModel(classes['label_id'], rgb_weights, flow_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model

    print("Set size: " + str(len(partition['test'])))

    # Load chunks
    test_splits = utils.make_chunks(original_list=partition['test'], size=2**15, chunk_size=2**12)

    rgb_dir = "/media/pedro/actv-ssd/foveated_" + params['gen_type'] + "_gc/"
    flow_dir = "test/flow/actv-ssd/flow_" + params['gen_type']

    test_chunks_count = 0
    with tf.device('/gpu:0'):
        for testIDS in test_splits:
            x_val_rgb, x_val_flow, y_val_pose, y_val_object, y_val_human = load_split(testIDS, params['dim'], params['n_channels'], 10, rgb_dir, flow_dir, train=False)
            print("Predicting on chunk " + str(test_chunks_count) + "/" + str(len(test_splits)) + ":")
            predictions = model.predict([x_val_rgb, x_val_flow], batch_size=params['batch_size'], verbose=1)
            utils.pred2classes(testIDS, predictions, output_csv)  # Convert predictions to readable output and perform majority voting
            test_chunks_count += 1

    if params['sendmail']:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com',
                        to_addr_list=['pedro_abreu95@hotmail.com', 'joaogamartins@gmail.com'],
                        subject='Finished ' + params['gen_type'] + ' prediction for two stream.',
                        message='Testing fusion with following params: ' + str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
