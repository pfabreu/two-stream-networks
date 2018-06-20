from keras.models import Model
from keras.layers import Input, Dense, Dropout


def context_create_model(classes, model_name="mlp", in_shape=(720,)):

    inputs = Input(shape=in_shape)

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    POSE_CLASSES = 14
    OBJ_HUMAN_CLASSES = 49
    HUMAN_HUMAN_CLASSES = 17
    pred_pose = Dense(POSE_CLASSES, activation='softmax',name='pred_pose')(x)
    pred_obj_human = Dense(OBJ_HUMAN_CLASSES, activation='sigmoid',name='pred_obj_human')(x)
    pred_human_human = Dense(HUMAN_HUMAN_CLASSES, activation='sigmoid', name='pred_human_human')(x)
    model = Model(inputs=inputs, outputs=[pred_pose, pred_obj_human, pred_human_human])
    
    return model


def compile_model(model):
    model.compile(optimizer='rmsprop', loss=['categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['categorical_accuracy'], loss_weights=[1.0, 1.0, 1.0])
    return model
