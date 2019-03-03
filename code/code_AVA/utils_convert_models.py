import smtplib
import csv
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import scipy.io as spio
from collections import Counter
import pickle

def create_caffe_pkl(stream = 'flow'):
    import caffe
    print("For conversation TSN, please use the custom caffe version")

    model_path = '../models/kinetics_caffe/inception_v3_kinetics_' + stream + '_pretrained/inception_v3_' + stream + '_deploy.prototxt'
    param_path = '../models/kinetics_caffe/inception_v3_kinetics_' + stream + '_pretrained/inception_v3_' + stream + '_kinetics.caffemodel'
    caffe.set_mode_cpu
    net = caffe.Net(model_path, param_path, caffe.TEST)

    save_name = '../models/kinetics_keras/tsn_%s_params' % stream
    order_name = '%s_names' % save_name
    # bn_order_name = 'bn_%s'%order_name
    dict_name = net.params
    # bn_names_order = [t for t in net.params.keys() if 'batchnorm' in t]
    param_dict = {}
    for key, value in net.params.items():
        param_dict[key] = [t.data for t in value]

    output = open(save_name + '.pkl', 'wb')
    pickle.dump(param_dict, output)
    output.close()

    output = open(order_name + '.pkl', 'wb')
    pickle.dump(net.params.keys(), output)
    output.close()
    # save_obj(bn_names_order, bn_order_name)


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    layer_weights = []
    d = data['net']
    elem = d.__dict__["params"]
    for e in elem:
        e = _todict(e)
        layer_weights.append(e)
    return layer_weights

def convert_vgg(model, ucf_weights):
    """
    Converts VGG MatConvNet model to keras
    """
    ucf_w_count = 0  # bottom_layer_count = 35  # Number of keras layers until the top (fc layers) is reached, in vgg16 its 35
    for layer in model.layers:
        config = layer.get_config()
        name = config['name']
        # if ucf_w_count < bottom_layer_count:
        if name[:4] == 'conv':  # If its a convolutional layer
            print("Keras layer name: " + str(name))
            w = ucf_weights[ucf_w_count]['value']
            ucf_w_count += 1
            conv_w = np.asarray(w)
            print("\t MConvNet conv: " + str(conv_w.shape))
            w = ucf_weights[ucf_w_count]['value']
            bias_w = np.asarray(w)
            print("\t MConvNet bias: " + str(bias_w.shape))
            keras_weights = []
            keras_weights.append(conv_w)
            keras_weights.append(bias_w)
            ucf_w_count += 1
            # Read old weights
            old_weights = layer.get_weights()[0]
            print("\t Keras conv: " + str(old_weights.shape))
            old_biases = layer.get_weights()[1]
            print("\t Keras bias: " + str(old_biases.shape))
            # Load weights if shapes match (joao this is just me being ocd)
            if (old_weights - conv_w).all() and (old_biases - bias_w).all:
                layer.set_weights(keras_weights)


def convert_resnet(model, ucf_weights):
    """
    Converts ResNet MatConvNet model to keras
    """
    for layer in model.layers:
        config = layer.get_config()
        name = config['name']
        if name[:3] == 'res' or name[:4] == 'conv':  # If its a convolutional layer
            print("Keras layer name: " + str(name))
            eq_layer_num = 0
            for mlayer in ucf_weights:
                lname = mlayer['name']
                mlname = lname.rsplit('_', 1)[0]
                if mlname == name:
                    print(lname)
                    break
                eq_layer_num += 1
            print(eq_layer_num)
            w = ucf_weights[eq_layer_num]['value']
            conv_w = np.array(w, ndmin=4)  # This is needed
            print("\t MConvNet conv: " + str(conv_w.shape))
            eq_layer_num += 1
            w = ucf_weights[eq_layer_num]['value']
            bias_w = np.asarray(w)
            print("\t MConvNet bias: " + str(bias_w.shape))
            keras_weights = []
            keras_weights.append(conv_w)
            keras_weights.append(bias_w)
            layer.set_weights(keras_weights)

        elif name[:2] == 'bn':  # If it's a batch normalization layer
            print("Keras layer name: " + str(name))
            eq_layer_num = 0
            for mlayer in ucf_weights:
                lname = mlayer['name']
                mlname = lname.rsplit('_', 1)[0]
                if mlname == name:
                    break
                eq_layer_num += 1

            w = ucf_weights[eq_layer_num]['value']
            gamma = np.asarray(w)
            eq_layer_num += 1
            print("\t MConvNet bn: " + str(gamma.shape))
            w = ucf_weights[eq_layer_num]['value']
            beta = np.asarray(w)
            eq_layer_num += 1
            print("\t MConvNet bn: " + str(beta.shape))
            w = ucf_weights[eq_layer_num]['value']
            moments = np.asarray(w)
            print("\t MConvNet bn: " + str(moments.shape))

            keras_weights = []
            keras_weights.append(gamma)
            keras_weights.append(beta)
            keras_weights.append(moments[:, 0])
            keras_weights.append(moments[:, 1])
            layer.set_weights(keras_weights)


def convert_inceptionv3(model, keras_weights, keras_layer_names):
    keras_layer_names = []
    with open(keras_weights[0], "rb") as names_file:
        layer_names = pickle.load(names_file)

    with open(keras_weights[1], "rb") as params_file:
        layer_params = pickle.load(params_file)

    #for layer in model.layers:
    #    config = layer.get_config()
    #    name = config['name']
    #    if name[:4] == 'conv':
    #        layer.use_bias = True

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
