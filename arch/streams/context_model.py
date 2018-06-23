from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.models import Model
import utils


def context_create_model(NHU1, NHU2, in_shape=(480,)):

    inputs = Input(shape=in_shape)

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(NHU1, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(NHU2, activation='relu')(x)
    x = Dropout(0.5)(x)

    pred_pose = Dense(utils.POSE_CLASSES, activation='softmax')(x)
    pred_obj_human = Dense(utils.OBJ_HUMAN_CLASSES, activation='sigmoid')(x)
    pred_human_human = Dense(utils.HUMAN_HUMAN_CLASSES, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=[pred_pose, pred_obj_human, pred_human_human])
    return model


def compile_model(model):
    model.compile(optimizer='rmsprop', loss=['categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['categorical_accuracy'], loss_weights=[1.0, 1.0, 1.0])
    return model
