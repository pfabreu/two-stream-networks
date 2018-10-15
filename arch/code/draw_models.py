import os
CPU = False
if CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # This must be imported before keras

from context_data import load_split, get_AVA_set, get_AVA_labels
from context_lstm_model import context_create_modelA, context_create_modelB
from context_mlp_model import context_create_model
from rgb_model import rgb_create_model
from flow_model import flow_create_model
from two_stream_model import TwoStreamModel
from fusion_context_model import FusionContextModel
from pose_model import pose_create_model

import pickle
import numpy as np
from keras.callbacks import ModelCheckpoint
import utils
import itertools
from keras.layers import concatenate
import sys
from keras.utils import to_categorical
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from keras import backend as K

def main():
    K.clear_session()
    modelname = "fusion-lstm"
    if modelname == "lstmA":
        model = context_create_modelA(128, 64, 5, 30 * 3)
    elif modelname == "lstmB":
        model = context_create_modelB(128, 64, 5, 30 * 3)
    elif modelname == "mlp":
        model = context_create_model(128, 64, in_shape=(270,))
    elif modelname == "rgb":
        model, keras_layer_names = rgb_create_model(classes=None, soft_sigmoid=True, model_name='resnet50', freeze_all=True, conv_fusion=False)
    elif modelname == "flow":
        model, keras_layer_names = flow_create_model(classes=None, model_name='resnet50', soft_sigmoid=True, image_shape=(224, 224), opt_flow_len=20, freeze_all=True, conv_fusion=False)
    elif modelname == "two-stream":
        rgb_weights = "rgb_gauss_resnet50_1806290918.hdf5"
        flow_weights = "flow_resnet50_1806281901.hdf5"
        nsmodel = TwoStreamModel(None, rgb_weights, flow_weights)
        model = nsmodel.model
    elif modelname == "fusion-mlp":
        rgb_weights = "rgb_gauss_resnet50_1806290918.hdf5"
        flow_weights = "flow_resnet50_1806281901.hdf5"
        context_weights = "bestModelContext_128.hdf5"
        nsmodel = FusionContextModel(None, rgb_weights, flow_weights, context_weights)
        model = nsmodel.model
    
    elif modelname == "fusion-lstm":
        rgb_weights = "rgb_gauss_resnet50_1806290918.hdf5"
        flow_weights = "flow_resnet50_1806281901.hdf5"
        context_weights = None
        nsmodel = FusionContextModel(None, rgb_weights, flow_weights, None)
        model = nsmodel.model
    elif modelname == "pose":
        model = pose_create_model(None, 'alexnet', soft_sigmoid=True, image_shape=(300, 300), freeze_all=False, conv_fusion=False)





    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['categorical_accuracy'])
    plot_model(model, to_file=modelname + '.png', show_shapes=True, show_layer_names=True)

    sys.exit(0)


if __name__ == '__main__':
    main()
