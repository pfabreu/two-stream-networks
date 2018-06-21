from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.models import Model, Sequential
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
import utils


def print_params(model):
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))


def compile_model(model, soft_sigmoid=False):
    lw = [1.0, 1.0, 1.0]
    if soft_sigmoid is False:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics=['categorical_accuracy'], loss_weights=lw)
    return model


def rgb_create_model(classes, soft_sigmoid=False, model_name='inceptionv3', freeze_all=True):
    conv_fusion = False
    # TODO Make this multi-GPU
    with tf.device('/gpu:0'):
        if model_name == "resnet50":
            base_model = ResNet50(include_top=False, weights='imagenet', pooling=None, input_shape=(224, 224, 3))
        elif model_name == "inceptionv3":
            base_model = InceptionV3(include_top=False, weights='imagenet', pooling=None)
        elif model_name == "inception_resnet_v2":
            base_model = InceptionResNetV2(include_top=False, weights='imagenet', pooling=None)
        elif model_name == "shallow":
            base_model = Shallow(include_top=False, input_shape=(224, 224))

        # print base_model.summary()
        if conv_fusion is True:
            x = base_model.layers[-2].output
            x = Conv2D(64, (2, 2), name='ConvFusion')(x)
        else:
            x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)

        # Freeze layers
        if freeze_all is True:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            if model_name == "resnet50":
                # print base_model.summary
                for layer in base_model.layers[:160]:
                    layer.trainable = False
                for layer in base_model.layers[160:]:
                    layer.trainable = True
            elif model_name == "shallow":
                pass
        model = None
        if soft_sigmoid is False:
            predictions = Dense(len(classes), activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        else:
            pred_pose = Dense(utils.POSE_CLASSES, activation='softmax', name='pred_pose')(x)
            pred_obj_human = Dense(utils.OBJ_HUMAN_CLASSES, activation='sigmoid', name='pred_obj_human')(x)
            pred_human_human = Dense(utils.HUMAN_HUMAN_CLASSES, activation='sigmoid', name='pred_human_human')(x)
            model = Model(inputs=base_model.input, outputs=[pred_pose, pred_obj_human, pred_human_human])
        print_params(model)
        # print model.summary()
        print("Total number of layers in base model: " + str(len(base_model.layers)))
        print("Total number of layers in model: " + str(len(model.layers)))
    return model


def Shallow(include_top=False, input_shape=(224, 224)):
    # Architecture from the original paper of 2stream
    # http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf
    model = Sequential()

    model.add(Conv2D(96, (7, 7), strides=2, padding='same', name='conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # NOTE This is the only different layer between RGB and OF streams in the original paper
    model.add(Conv2D(256, (5, 5), strides=2, padding='same', name='conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', name='conv3'))
    model.add(Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', name='conv4'))
    model.add(Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', name='conv5'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # full6
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.9))

    # full7
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(1024, activation='relu'))

    return model
