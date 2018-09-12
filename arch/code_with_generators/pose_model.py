from keras.layers import Input, GlobalMaxPooling2D, AveragePooling2D, ZeroPadding2D, add, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model, Sequential
import utils


def pose_create_model(classes, model_name, image_shape=(224, 224)):
    input_shape = (image_shape[0], image_shape[1], 3)  # TODO More channels for pose
    base_model = AlexNet(input_shape)

    x = base_model.output

    # Pose model is trained from scratch
    pred_pose = Dense(utils.POSE_CLASSES, activation='softmax', name='logits')(x)
    model = Model(inputs=base_model.input, outputs=pred_pose)
    return model


def AlexNet(input_shape=(224, 224)):
    # Architecture from the original paper of 2stream
    # http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf

    # NOTE By default Keras uses glorot uniform which is Xavier initialization (use another if you feel like it)
    img_input = Input(shape=input_shape)

    x = Conv2D(96, (7, 7), strides=2, padding='same', name='conv1', kernel_initializer='he_uniform')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # NOTE This is the only different layer between RGB and OF streams in the original paper
    x = Conv2D(256, (5, 5), strides=2, padding='same', name='conv2', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', name='conv3', kernel_initializer='he_uniform')(x)
    x = Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', name='conv4', kernel_initializer='he_uniform')(x)
    x = Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', name='conv5', kernel_initializer='he_uniform')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # full6
    x = Flatten()(x)
    x = Dense(4096, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.9)(x)

    # full7
    x = Dense(2048, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.9)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_uniform')(x)

    model = Model(img_input, x, name='alexnet')
    return model
