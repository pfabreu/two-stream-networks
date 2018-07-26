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
    root_dir = '../../data/AVA/files/'

    # Erase previous models from GPU memory
    K.clear_session()

    # Load list of action classes and separate them
    # classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    checkpointer, tb, early_stopper = callbacks()

    # print len(classes['label-id'])
    # Parameters for training (batch size 32 is supposed to be the best)
    params = {'dim': (224, 224), 'batch_size': 512,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 10}

    generators = get_generators(classes=101, image_shape=params['dim'], batch_size=params['batch_size'])

    # Create + compile model, load saved weights if true
    rgb_weights = None

    # Create new model and freeze all but top
    model = rgb_create_model(model_name='resnet50', conv_fusion=True)

    if rgb_weights is None:

        # If they don't exist, convert Feichtenhofer's models from matconv to keras
        ucf_weights = None
        if ucf_weights is None:
                # TODO Better initialization, average UCF models overt he 3 splits provided
            ucf_weights = utils.loadmat("../models/ucf_matconvnet/ucf101-img-resnet-50-split1.mat")
            utils.convert_resnet(model, ucf_weights)
            model.save("../models/ucf_keras/keras-ucf101-rgb-resnet50.hdf5")
        else:
                # Load pre-trained UCF weights
            model.load_weights("../models/ucf_keras/keras-ucf101-rgb-resnet50.hdf5")

        # Freeze all but top
        for layer in model.layers[:-2]:
            layer.trainable = False

        # Compile the model
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        sys.exit(0)

        # Train a bit more
        model = train(model, nb_epochs, generators)  # No need for callbacks here
    else:

        # Create new model
        model = rgb_create_model(model_name='resnet50', conv_fusion=True)
        # Load weights
        model.load_weights(rgb_weights)

    # Unfreeze layers for a more complete training
    for layer in model.layers:
        layer.trainable = True

    # Recompile the model
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

    model = train(model, nb_epochs, generators, [checkpointer, tb, logger, early_stopper])


if __name__ == '__main__':
    main()
