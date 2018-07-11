import tensorflow as tf
import utils
from stream_2_model import TwoStreamModel
from stream_2_data import get_AVA_set, get_AVA_classes, load_split
import time
from keras import backend as K


def main():
    root_dir = '../../data/AVA/files/'
    K.clear_session()

    # Load list of action classes and separate them (from utils_stream)
    classes = get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': (224, 224), 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 150, 'model': 'resnet50', 'sendmail': True, 'gen_type': 'test'}

    # Get validation set from directory
    partition = {}
    partition['test'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Test_Custom_Corrected.csv", train=False)

    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    result_csv = "output_" + params['gen_type'] + "_" + time_str + ".csv"

    # Load trained model
    rgb_weights = "../models/rgb_resnet50_1805290059.hdf5"
    flow_weights = "../models/flow_resnet50_1805290120.hdf5"
    nsmodel = TwoStreamModel(classes['label_id'], rgb_weights, flow_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model
    two_stream_weights = "../models/two_stream_fusion_resnet50_1805290059.hdf5"
    model.load_weights(two_stream_weights)

    print("Test set size: " + str(len(partition['test'])))

    # Load chunks
    test_splits = utils.make_chunks(original_list=partition['test'], size=len(partition['test']), chunk_size=2**12)

    # Test directories where pre-processed test files are
    rgb_dir = "/media/pedro/actv-ssd/foveated_" + params['gen_type'] + "_gc/"
    flow_dir = "test/flow/actv-ssd/flow_" + params['gen_type']

    test_chunks_count = 0
    with tf.device('/gpu:0'):
        for testIDS in test_splits:
            x_test_rgb, x_test_flow, y_test_pose, y_test_object, y_test_human = load_split(testIDS, params['dim'], params['n_channels'], 10, rgb_dir, flow_dir, "test", "rgb", train=False)
            print("Predicting on chunk " + str(test_chunks_count) + "/" + str(len(test_splits)) + ":")
            # Convert predictions to readable output and perform majority voting
            predictions = model.predict([x_val_rgb, x_val_flow], batch_size=params['batch_size'], verbose=1)
            utils.pred2classes(testIDS, predictions, result_csv)
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
