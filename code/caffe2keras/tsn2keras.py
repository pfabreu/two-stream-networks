import pickle
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, Sequential
from keras.layers import Input, GlobalMaxPooling2D, AveragePooling2D, ZeroPadding2D, add, concatenate, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization


# Inception v3 has 11 mixed and 4 conv_bn's before mixeds
keras_layer_names = []


"""Inception V3 model for Keras.
Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).
# Reference
- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567)
"""


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    bn_axis = 3
    # NOTE use_bias had to be set to True here to match TSN
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=True, name=conv_name)(x)
    keras_layer_names.append(x.name.split("/")[0])
    # NOTE scale had to be set to True here to match TSN
    x = BatchNormalization(axis=bn_axis, scale=True, name=bn_name)(x)
    keras_layer_names.append(x.name.split("/")[0])
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3_TSN(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000):

    # Determine proper input shape

    img_input = Input(shape=input_shape)
    channel_axis = 3
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9, 10: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    model = Model(img_input, x, name='inception_v3')

    return model

GPU = False
CPU = True

num_cores = 4

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

stream_type = 'flow'
# Do this on the CPU because my GPU was busy :(
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                        device_count={'CPU': num_CPU, 'GPU': num_GPU})
session = tf.Session(config=config)
K.set_session(session)


with open("../models/kinetics_keras/tsn_" + stream_type + "_params_names.pkl", "rb") as names_file:
    layer_names = pickle.load(names_file)

with open("../models/kinetics_keras/tsn_" + stream_type + "_params.pkl", "rb") as params_file:
    layer_params = pickle.load(params_file)

print(layer_names)
# print(layer_params)

# Create keras inception v3 model and print layers
model = InceptionV3_TSN(include_top=False, weights='imagenet', pooling=None, input_shape=(224, 224, 10))
for layer in model.layers:
    config = layer.get_config()
    name = config['name']
    if name[:4] == 'conv':
        layer.use_bias = True

print(model.summary())
print(len(layer_params))
print(len(layer_params['conv_Conv2D']))  # seems to correspond to conv2d_1
print(len(layer_params['conv_batchnorm']))  # seems to correspond to batch_normalization_1 (but must include the activation layer keras has)

conv_layer = layer_params['conv_Conv2D'][0]
conv_layer_actv = layer_params['conv_Conv2D'][1]
bn_layer_gamma = layer_params['conv_batchnorm'][0]
bn_layer_beta = layer_params['conv_batchnorm'][1]
bn_layer_moments = layer_params['conv_batchnorm'][2]
bn_layer_actv = layer_params['conv_batchnorm'][3]
bn_layer_actv = np.squeeze(bn_layer_actv)

conv_layer = np.swapaxes(conv_layer, 3, 0)

print(conv_layer.shape)
print(conv_layer_actv.shape)
print(bn_layer_gamma.shape)
print(bn_layer_beta.shape)
print(bn_layer_moments.shape)
print(bn_layer_actv.shape)

l = 0
# for layer in model.layers:
for ln in keras_layer_names:
    layer = model.get_layer(ln)
    config = layer.get_config()
    name = config['name']
    if name[:4] == 'conv':
        print("\t Caffe name: " + layer_names[l])
        print("\t Keras name: " + name)
        # print(len(layer.get_weights()))
        old_weights = layer.get_weights()[0]
        caffe_conv = layer_params[layer_names[l]][0]
        caffe_conv = np.swapaxes(caffe_conv, 3, 0)
        caffe_conv = np.swapaxes(caffe_conv, 1, 2)
        print("\t Keras conv: " + str(old_weights.shape))
        print("\t Caffe conv: " + str(caffe_conv.shape))

        old_biases = layer.get_weights()[1]
        caffe_bias = layer_params[layer_names[l]][1]
        print("\t Keras bias: " + str(old_biases.shape))
        print("\t Caffe bias: " + str(caffe_bias.shape) + "\n")

        keras_weights = []
        keras_weights.append(caffe_conv)
        keras_weights.append(caffe_bias)
        layer.set_weights(keras_weights)

        l += 1
    elif name[:5] == 'batch':
        print("\t Caffe name: " + layer_names[l])
        print("\t Keras name: " + name)
        # print(len(layer.get_weights()))
        old_weights = layer.get_weights()[0]
        caffe_bn_gamma = layer_params[layer_names[l]][0]
        caffe_bn_gamma = np.squeeze(caffe_bn_gamma)
        print("\t Keras bn gamma: " + str(old_weights.shape))
        print("\t Caffe bn gamma: " + str(caffe_bn_gamma.shape))
        old_weights = layer.get_weights()[1]
        caffe_bn_beta = layer_params[layer_names[l]][1]
        caffe_bn_beta = np.squeeze(caffe_bn_beta)
        print("\t Keras bn beta: " + str(old_weights.shape))
        print("\t Caffe bn beta: " + str(caffe_bn_beta.shape))
        old_weights = layer.get_weights()[2]
        caffe_bn_moments = layer_params[layer_names[l]][2]
        caffe_bn_moments = np.squeeze(caffe_bn_moments)
        print("\t Keras bn moments: " + str(old_weights.shape))
        print("\t Caffe bn moments: " + str(caffe_bn_moments.shape))
        old_weights = layer.get_weights()[3]
        caffe_bn_scale = layer_params[layer_names[l]][3]
        caffe_bn_scale = np.squeeze(caffe_bn_scale)
        print("\t Keras bn scale: " + str(old_weights.shape))
        print("\t Caffe bn_scale: " + str(caffe_bn_scale.shape) + "\n")

        keras_weights = []
        keras_weights.append(caffe_bn_gamma)
        keras_weights.append(caffe_bn_beta)
        keras_weights.append(caffe_bn_moments)
        keras_weights.append(caffe_bn_scale)
        layer.set_weights(keras_weights)

        l += 1
    elif name[:5] == 'activ':
        pass
        # print("\t Keras activations (for BN): ")
        # print(len(layer.get_weights()))
    elif name[:5] == 'avera' or name[:3] == 'max':
        pass
        # print("\t Keras pooling: ")
        # print(len(layer.get_weights()))
    elif name[:5] == 'mixed':
        pass
        # print(len(layer.get_weights()))
        # print("\t Mixed layer ")
model.save("../models/kinetics_keras/keras-tsn-kinetics-flow-inception-v3-notop.hdf5")
