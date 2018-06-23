# Runs a single sample through the model
import tensorflow as tf
import utils
from stream_2_model import NStreamModel
from stream_2_data import get_AVA_set, get_AVA_classes, load_split
import time


def main():

    # Load list of action classes and separate them (from utils_stream)
    classes = get_AVA_classes('AVA2.1/ava_action_list_v2.1.csv')

    # Create + compile model, load saved weights if they exist
    rgb_weights = "models/rgb_resnet50_1805290059.hdf5"
    flow_weights = "models/flow_resnet50_1805290120.hdf5"
    nsmodel = NStreamModel(classes['label_id'], rgb_weights, flow_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model

    rgb_dir = "/media/pedro/actv-ssd/foveated_" + params['gen_type'] + "_gc/"
    flow_dir = "test/flow/actv-ssd/flow_" + params['gen_type']

    with tf.device('/gpu:0'):
        x_rgb, x_flow = load_sample(testIDS, params['dim'], params['n_channels'], 10, rgb_dir, flow_dir, train=False)
        prediction = model.predict([x_val_rgb, x_val_flow], batch_size=params['batch_size'], verbose=1)
        # TODO Convert predictions to readable output and perform majority voting



if __name__ == '__main__':
    main()
