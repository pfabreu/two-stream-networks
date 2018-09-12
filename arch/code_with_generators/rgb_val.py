import warnings
import os
import time
import itertools
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from keras import backend as K
from rgb_model import create_model
from rgb_data import get_classes, get_generators
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard


def main():
    # Erase previous models from GPU memory
    root_dir = '../../data/UCF101/files/'

    # Erase previous models from GPU memory
    # K.clear_session()

    # Load list of action classes and separate them
    classes, data_list_clean, val_generator, steps = get_classes_val(root_dir + 'data_list.csv')

    rgb_weights = None

    # Create new model and freeze all but top
    model, keras_layer_names = rgb_create_model(model_name='resnet50', conv_fusion=False)
    model.load(rgb_weights)

    # Evaluate the model!
    # results = model.evaluate_generator(generator=val_generator, steps=steps)
    # print(results)
    # print(model.metrics_names)

    model.fit_generator(generator=val_generator, steps_per_epoch=steps, max_queue_size=1)
    print('Finished validation of weights:', rgb_weights)

if __name__ == '__main__':
    main()
