import warnings
import os
import time
import itertools
import tensorflow as tf
import gc
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

# Disable two annoying warnings: one from hdf5 (due to np ver) and another
# from tf not built with AVX/FMA
#warnings.simplefilter(action='ignore', category=FutureWarning)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from rgb_training_model import create_model, compile_model
from rgb_training_data import load_split, DataGenerator, get_AVA_classes, get_AVA_set, get_AVA_labels
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
# warnings.resetwarnings()


def callbacks():
    # Callbacks(besides the basic ones used by Keras by default)
    # -- Callback: Model checkpoint (saves checkpoints)
    time_now = time.strftime("%y%m%d%H%M", time.localtime())
    ckpt_dir = os.path.join(
        'out', "rgb-pedro-desktop-" + time_now, 'checkpoints')
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


def main():
    checkpointer, tb, logger, early_stopper = callbacks()

    # Load list of action classes and separate them (from utils_stream)
    classes = get_AVA_classes('AVA2.1/ava_action_list_v2.1.csv')
    # print len(classes['label-id'])
    # Parameters for training (batch size 32 is supposed to be the best)
    params = {'dim': (224, 224), 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 150}

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['train'] = get_AVA_set(
        classes=classes, filename="AVA2.1/ava_mini_split_train.csv",
        dir="/ava_train_resized/rgb/")  # IDs for training

    partition['validation'] = get_AVA_set(
        classes=classes, filename="AVA2.1/ava_mini_split_validation.csv",
        dir="/ava_val_resized/rgb/")  # IDs for validation

    # Labels
    labels_train = get_AVA_labels(classes, partition, "train")
    labels_val = get_AVA_labels(classes, partition, "validation")
    # Create + compile model, load saved weights if true
    saved_weights = None
    if saved_weights is None:
        model = create_model(classes=classes['label_id'], soft_sigmoid=False)
        model = compile_model(model, soft_sigmoid=False)
    else:
        # TODO Load weights
        pass

    # Load first 6000 of partition{'train'}
    seq = partition['train'][:16384]
    train_splits = [seq[i:i + 4096] for i in range(0, len(seq), 4096)]

    maxValAcc = 0.0
    bestModelPath = "bestModel.hdf5"
    with tf.device('/gpu:0'):
        for epoch in range(params['nb_epochs']):
            for trainIDS in train_splits:
                # Load x_train
                x_train, y_train = load_split(trainIDS, labels_train, params['dim'], params['n_channels'], "train")
                y_train = to_categorical(y_train, num_classes=params['n_classes'])
                model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=1, verbose=1)
            # Load val_data
            x_val, y_val = load_split(partition['validation'][:4096], labels_val, params['dim'], params['n_channels'], "val")
            y_val = to_categorical(y_val, num_classes=params['n_classes'])
            loss, acc = model.evaluate(x_val, y_val, batch_size=params['batch_size'])
            if acc > maxValAcc:
                model.save(bestModelPath)
                maxValAcc = acc


if __name__ == '__main__':
    main()
