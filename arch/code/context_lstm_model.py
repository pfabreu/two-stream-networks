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


def context_create_modelB(NHU1, NHU2, timewindow, n_features):
    past_input = Input(shape=(timewindow + 1, n_features))
    past_model = Bidirectional(LSTM(NHU1, return_sequences=False, kernel_initializer='he_uniform'))(past_input)

    future_input = Input(shape=(timewindow + 1, n_features))
    future_model = Bidirectional(LSTM(NHU1, return_sequences=False, go_backwards=True, kernel_initializer='he_uniform'))(future_input)

    merge_model = concatenate([past_model, future_model])
    merge_model = Dense(NHU2, activation='relu', kernel_initializer='he_uniform')(merge_model)
    merge_model = Dropout(0.5)(merge_model)
    pred_pose = Dense(utils.POSE_CLASSES, activation='softmax', name='pred_pose')(merge_model)
    pred_obj_human = Dense(utils.OBJ_HUMAN_CLASSES, activation='sigmoid', name='pred_obj_human')(merge_model)
    pred_human_human = Dense(utils.HUMAN_HUMAN_CLASSES, activation='sigmoid', name='pred_human_human')(merge_model)
    m = concatenate([pred_pose, pred_obj_human, pred_human_human, ], axis=-1)
    model = Model(inputs=[past_input, future_input], outputs=[pred_pose, pred_obj_human, pred_human_human])
    return model


def context_create_modelA(NHU1, NHU2, timewindow, n_features):
    pastInput = Input(shape=(timewindow + 1, n_features))
    futureInput = Input(shape=(timewindow + 1, n_features))

    past = LSTM(NHU1, activation="tanh", use_bias=True, return_sequences=True)(pastInput)
    future = LSTM(NHU1, activation="tanh", use_bias=True, return_sequences=True, go_backwards=True)(futureInput)
    x = concatenate([past, future])

    x = LSTM(NHU2, activation="tanh", use_bias=True, return_sequences=False)(x)
    pred_pose = Dense(utils.POSE_CLASSES, activation='softmax')(x)
    pred_obj_human = Dense(utils.OBJ_HUMAN_CLASSES, activation='sigmoid')(x)
    pred_human_human = Dense(utils.HUMAN_HUMAN_CLASSES, activation='sigmoid')(x)
    model = Model(inputs=[pastInput, futureInput], outputs=[pred_pose, pred_obj_human, pred_human_human])
    return model
