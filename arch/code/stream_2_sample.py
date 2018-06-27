"""
 Runs a single sample through the model
"""

import tensorflow as tf
from stream_2_model import TwoStreamModel
from stream_2_data import get_AVA_classes, load_sample


def main():
    # 3s segment
    segment = "GBXK_SyfisM_1046.avi"

    # TODO Extract BB's (if not done already, otherwise we provide them here)

    # TODO Extract flow (if not done already)

    # TODO Extract rgb's with filter (gaussian, crop, fovea)

    # Load list of action classes and separate them (from utils_stream)
    classes = get_AVA_classes('AVA2.1/ava_action_list_v2.1.csv')

    # Create + compile model, load saved weights if they exist
    rgb_weights = "models/rgb_resnet50_1805290059.hdf5"
    flow_weights = "models/flow_resnet50_1805290120.hdf5"
    nsmodel = TwoStreamModel(classes['label_id'], rgb_weights, flow_weights)
    nsmodel.compile_model(soft_sigmoid=True)
    model = nsmodel.model

    rgb_dir = "/media/pedro/actv-ssd/foveated_" + params['gen_type'] + "_gc/"
    flow_dir = "test/flow/actv-ssd/flow_" + params['gen_type']

    batch_size = 32
    with tf.device('/gpu:0'):
        x_rgb, x_flow = load_sample(sampleIDS, (224, 224), rgb_dir, flow_dir)
        prediction = model.predict([x_rgb, x_flow], batch_size=batch_size, verbose=1)
        # TODO Convert predictions to readable output and perform majority voting


if __name__ == '__main__':
    main()
