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


def callbacks():
    # Callbacks(besides the basic ones used by Keras by default)
    # -- Callback: Model checkpoint (saves checkpoints)
    time_now = time.strftime("%y%m%d%H%M", time.localtime())
    ckpt_dir = os.path.join('out', "rgb-pedro-desktop-" + time_now, 'checkpoints')
    ckpt_file = os.path.join(ckpt_dir, '{epoch:03d}-{val_loss:.3f}.hdf5')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    checkpointer = ModelCheckpoint(filepath=ckpt_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, period=1)

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
    GPU = False
    CPU = True
    num_cores = 8

    if GPU:
        num_GPU = 1
        num_CPU = 1
    if CPU:
        num_CPU = 1
        num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores, allow_soft_placement=True, device_count={'CPU': num_CPU, 'GPU': num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)

    # Erase previous models from GPU memory
    root_dir = '../../data/UCF101/files/'

    # Erase previous models from GPU memory
    # K.clear_session()

    # Load list of action classes and separate them
    classes, data_list_clean = get_classes(root_dir + 'data_list.csv')

    checkpointer, tb, early_stopper = callbacks()

    # print len(classes['label-id'])
    # Parameters for training (batch size 32 is supposed to be the best)
    params = {'dim': (224, 224), 'batch_size': 512, 'n_classes': 101, 'model': 'resnet50', 'n_channels': 3, 'shuffle': False, 'nb_epochs': 10}

    generators = get_generators(classes=classes, image_shape=params['dim'], batch_size=params['batch_size'])

    # Create + compile model, load saved weights if true
    rgb_weights = None

    # Create new model and freeze all but top
    model, keras_layer_names = rgb_create_model(model_name='resnet50', conv_fusion=False)

    # TODO Experiment: 1. no initialization, 2. ucf initialization 3. kinetics initialization
    initialization = True  # Set to True to use initialization
    kinetics_weights = None
    ucf_weights = ""

    if rgb_weights is None:
        # If they don't exist, convert Feichtenhofer's models from matconv to keras
        if initialization is True:
            if ucf_weights is None:
                # TODO Better initialization, average UCF models overt he 3 splits provided
                if params['model'] == "resnet50":
                    ucf_weights = utils.loadmat("../models/ucf_matconvnet/ucf101-img-resnet-50-split1.mat")
                    utils.convert_resnet(model, ucf_weights)
                    model.save("models/keras-ucf101-rgb-resnet50.hdf5")
            else:
                if ucf_weights != "":
                    # Load pre-trained UCF weights
                    model.load_weights("../models/ucf_keras/keras-ucf101-rgb-resnet50.hdf5")

            if kinetics_weights is None:
                if params['model'] == "inceptionv3":
                    keras_weights = ["../models/kinetics_keras/tsn_rgb_params_names.pkl", "../models/kinetics_keras/tsn_rgb_params.pkl"]
                    utils.convert_inceptionv3(model, keras_weights, keras_layer_names)
                    model.save("models/keras-kinetics-rgb-inceptionv3.hdf5")
            else:
                if kinetics_weights != "":
                    model.load_weights("models/keras-kinetics-rgb-inceptionv3.hdf5")

        for layer in model.layers[:-2]:
            layer.trainable = False

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train a bit more
        model = train(model, nb_epochs, generators)  # No need for callbacks here
    else:
        # Create new model
        model = rgb_create_model(model_name='resnet50', conv_fusion=False)
        # Load weights
        model.load_weights(rgb_weights)

    # Unfreeze layers for a more complete training
    for layer in model.layers[:(len(model.layers) // 2)]:
        layer.trainable = False
    for layer in model.layers[(len(model.layers) // 2):]:
        layer.trainable = True

    # Recompile the model
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

    model = train(model, nb_epochs, generators, [checkpointer, tb, logger, early_stopper])


if __name__ == '__main__':
    main()
