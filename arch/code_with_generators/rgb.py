import warnings
import os
import time
import itertools
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from keras import backend as K
from rgb_model import create_model, compile_model
from rgb_data import load_split, DataGenerator, get_AVA_classes, get_AVA_set, get_AVA_labels
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard


def callbacks():
    # Callbacks(besides the basic ones used by Keras by default)
    # -- Callback: Model checkpoint (saves checkpoints)
    time_now = time.strftime("%y%m%d%H%M", time.localtime())
    ckpt_dir = os.path.join('out', "rgb-pedro-desktop-" + time_now, 'checkpoints')
    ckpt_file = os.path.join(ckpt_dir, '{epoch:03d}-{val_loss:.3f}.hdf5')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    checkpointer = ModelCheckpoint(
        filepath=ckpt_file, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, period=1)

    # -- Callback: Tensorboard
    tb_dir = os.path.join('out', "rgb-pedro-desktop-" + time_now, 'tb')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    tb = TensorBoard(log_dir=os.path.join(tb_dir))

    # -- Callback: Early stopper (might get lucky)
    early_stopper = EarlyStopping(monitor='loss', patience=100)

    return checkpointer, tb, early_stopper


def train(model, nb_epochs, generators, callbacks=[]):
    train_gen, val_gen = generators
    model.fit_generator(train_gen, steps_per_epoch=100, validation_data=val_gen, validation_steps=10, epochs=nb_epochs, callbacks=callbacks)
    return model


def main():

    # Erase previous models from GPU memory
    K.clear_session()

    checkpointer, tb, early_stopper = callbacks()

    # print len(classes['label-id'])
    # Parameters for training (batch size 32 is supposed to be the best)
    params = {'dim': (224, 224), 'batch_size': 512,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 10}

    data = DataSet(image_shape=params['dim'])

    generators = get_generators(data=data, image_shape=params['dim'], batch_size=params['batch_size'])

    # Create + compile model, load saved weights if true
    rgb_weights = None

    if rgb_weights is None:
        # Create new model and freeze all but top
        model = rgb_create_model(model_name='resnet50', freeze_all=True, conv_fusion=True):

            # Load pre-trained UCF weights
        model.load_weights("../models/ucf_keras/")

        model = train(model, nb_epochs, generators)  # No need for callbacks here

        # Now, freeze half layers for final training step
    else:

        # Create new model and freeze half
        model = rgb_create_model(model_name='resnet50', freeze_all=False, conv_fusion=True):

            # TODO Load weights
        model.load_weights(rgb_weights)

    model = train(model, nb_epochs, generators, [checkpointer, tb, logger, early_stopper])


if __name__ == '__main__':
    main()
