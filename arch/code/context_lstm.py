from context_data import load_split, get_AVA_set, get_AVA_labels
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
import tensorflow as tf
from keras import backend as K


def reshapeX(x_train, timesteps, features):
    ln = x_train.shape[0]
    i = 0
    print(ln)
    X_past = np.zeros([ln, (timesteps // 2) + 1, features])
    X_future = np.zeros([ln, (timesteps // 2) + 1, features])
    for xline in x_train:
        xline = np.split(xline, timesteps)
        t = 0
        for xtime in xline[:((timesteps // 2) + 1)]:
            X_past[i, t, ] = xtime
            t += 1

        t = 0
        for xtime in xline[(timesteps // 2):]:
            X_past[i, t, ] = xtime
            t += 1
        i += 1
    print(X_past.shape)
    print(X_future.shape)
    return X_past, X_future


def reshapeY(y_t, features):
    # Convert y_t to a single sequence
    ln = y_t[0].shape[0]
    i = 0
    print(ln)
    # y = np.zeros([ln, 1, features])
    y = np.zeros([ln, features])
    for yp, yo, th in itertools.izip(y_t[0], y_t[1], y_t[2]):

        yline = np.concatenate((yp, yo, th), axis=0)
        # y[i, 0, ] = yline
        y[i, ] = yline
    print(y.shape)
    return y


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

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})
    K.clear_session()
    session = tf.Session(config=config)
    K.set_session(session)

    root_dir = '../../data/AVA/files/'
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    partition = {}

    # Labels
    timewindow = 10  # NOTE To the past and to the future
    neighbours = 3
    num_classes = len(classes['label_name'])
    n_features = num_classes * neighbours
    context_dim = num_classes * neighbours * (timewindow + 1 + timewindow)

    # Load train data
    Xfilename = "XContext_train_tw10_n3.csv"
    partition['train'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Train_Custom_Corrected.csv")  # IDs for training
    labels_train = get_AVA_labels(classes, partition, "train", filename=root_dir + "AVA_Train_Custom_Corrected.csv")
    x_train, y_train_pose, y_train_object, y_train_human = load_split(partition['train'], labels_train, context_dim, 1, "train", Xfilename)
    y_t = []
    y_t.append(to_categorical(y_train_pose, num_classes=utils.POSE_CLASSES))
    y_t.append(utils.to_binary_vector(y_train_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
    y_t.append(utils.to_binary_vector(y_train_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))
    x_train_past, x_train_future = reshapeX(x_train, (timewindow + 1 + timewindow), n_features)
    # y_train = reshapeY(y_t, num_classes)

    # Load val data
    Xfilename = "XContext_val_tw10_n3.csv"
    partition['validation'] = get_AVA_set(classes=classes, filename=root_dir + "AVA_Val_Custom_Corrected.csv")  # IDs for training
    labels_val = get_AVA_labels(classes, partition, "validation", filename=root_dir + "AVA_Val_Custom_Corrected.csv")
    x_val, y_val_pose, y_val_object, y_val_human = load_split(partition['validation'], labels_val, context_dim, 1, "validation", Xfilename)
    y_v = []
    y_v.append(to_categorical(y_val_pose, num_classes=utils.POSE_CLASSES))
    y_v.append(utils.to_binary_vector(y_val_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
    y_v.append(utils.to_binary_vector(y_val_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))
    x_val_past, x_val_future = reshapeX(x_val, (timewindow + 1 + timewindow), n_features)
    # y_val = reshapeY(y_v, num_classes)

    # NOTE This is a hack, training on the actual validation (but chunk will be removed)
    x_train_past = np.vstack((x_train_past, x_val_past))
    x_train_future = np.vstack((x_train_future, x_val_future))
    y_t[0] = np.vstack((y_t[0], y_v[0]))
    y_t[1] = np.vstack((y_t[1], y_v[1]))
    y_t[2] = np.vstack((y_t[2], y_v[2]))
    # y_train = np.vstack((y_train, y_val))

    bestModelPath = "context_lstm.hdf5"
    histPath = "contextHistory"
    checkpointer = ModelCheckpoint(filepath=bestModelPath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, period=1)

    # Create LSTM
    # Had we stacked three recurrent hidden layers, we'd have set return_sequence=True to the second hidden layer and return_sequence=False to the last.
    # In other words, return_sequence=False is used as an interface from recurrent to feedforward layers (dense or convolutionnal).

    # model = Sequential()
    # model.add(Bidirectional(LSTM(512, input_shape=(21, 30), return_sequences=False)))
    # model.add(Dense(30, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam')

    # x = Input(shape=(21, 30))
    # model = Bidirectional(LSTM(512, return_sequences=False))(x)
    # pred_pose = Dense(utils.POSE_CLASSES, activation='softmax', name='pred_pose')(model)
    # pred_obj_human = Dense(utils.OBJ_HUMAN_CLASSES, activation='sigmoid', name='pred_obj_human')(model)
    # pred_human_human = Dense(utils.HUMAN_HUMAN_CLASSES, activation='sigmoid', name='pred_human_human')(model)
    # m = concatenate([pred_pose, pred_obj_human, pred_human_human, ], axis=-1)
    # model = Model(x, outputs=m)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

    past_input = Input(shape=(timewindow + 1, n_features))
    past_model = Bidirectional(LSTM(128, return_sequences=False, kernel_initializer='he_uniform'))(past_input)

    future_input = Input(shape=(timewindow + 1, n_features))
    future_model = Bidirectional(LSTM(128, return_sequences=False, kernel_initializer='he_uniform'))(future_input)

    merge_model = concatenate([past_model, future_model])
    merge_model = Dense(2 * n_features, activation='relu', kernel_initializer='he_uniform')(merge_model)
    merge_model = Dropout(0.5)(merge_model)
    pred_pose = Dense(utils.POSE_CLASSES, activation='softmax', name='pred_pose')(merge_model)
    pred_obj_human = Dense(utils.OBJ_HUMAN_CLASSES, activation='sigmoid', name='pred_obj_human')(merge_model)
    pred_human_human = Dense(utils.HUMAN_HUMAN_CLASSES, activation='sigmoid', name='pred_human_human')(merge_model)
    m = concatenate([pred_pose, pred_obj_human, pred_human_human, ], axis=-1)
    model = Model(inputs=[past_input, future_input], outputs=[pred_pose, pred_obj_human, pred_human_human])
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['categorical_accuracy'])

    # Train
    n_epoch = 200
    hist = model.fit([x_train_past, x_train_future], y_t, shuffle=False, validation_split=0.1, epochs=n_epoch, verbose=2, callbacks=[checkpointer])

    # model.save(bestModelPath)
    with open(histPath, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)

    # evaluate
    # result = model.predict(X, batch_size=n_batch, verbose=0)
    # for value in result[0,:,0]:
    #	print('%.1f' % value)


if __name__ == '__main__':
    main()
