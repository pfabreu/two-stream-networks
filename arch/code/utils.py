import smtplib
import csv
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import scipy.io as spio
from collections import Counter
import pickle

POSE_CLASSES = 10
OBJ_HUMAN_CLASSES = 12
HUMAN_HUMAN_CLASSES = 8


def decideBestContextModel(pickle_dir):
    lossList = []
    NHU = [32, 64, 128, 256, 512]
    for i in NHU:
        with open(pickle_dir + 'contextHistory_' + str(i), 'rb') as pickle_file:
            content = pickle.load(pickle_file)
            valloss = content['val_loss']
            lossList.append(min(valloss))


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


def get_AVA_classes(csv_filename):
    """
    Gets all classes from an AVA csv, format of classes is a dictionary with:
    classes['label_id'] has all class ids from 1-80
    classes['label_name'] has all class names (e.g bend/bow (at the waist))
    classes['label_type'] is either PERSON_MOVEMENT (1-14), OBJECT_MANIPULATION
    (15-63) or PERSON_INTERACTION (64-80)
    """
    classes = []
    with open(csv_filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        headers = next(csvReader)
        classes = {}
        for h in headers:
            classes[h] = []

        for row in csvReader:
            for h, v in zip(headers, row):
                classes[h].append(v)
    return classes


def sendemail(from_addr, to_addr_list, subject, message, login, password, smtpserver='smtp.gmail.com:587'):
    header = 'From: %s\n' % from_addr
    cc_addr_list = []
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    header += 'Subject: %s\n\n' % subject
    message = header + message

    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login, password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems


def make_chunks(original_list, size, chunk_size):
    seq = original_list[:size]
    splits = [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]
    return splits


def to_binary_vector(list_classes, size, labeltype):
    """
    Converts list_classes list to binary vector with given size
    This is specific to the AVA challenge
    """
    labelsarray = np.empty([len(list_classes), size])
    offset = 0
    if labeltype == 'object-human':
        offset = POSE_CLASSES
    elif labeltype == 'human-human':
        offset = POSE_CLASSES + OBJ_HUMAN_CLASSES
    elif labeltype == 'pose':
        offset = 0
    index = 0
    for l in list_classes:
        bv = np.zeros(size)
        lv = l
        if len(lv) >= 1:
            lv = [x - offset for x in lv]
        for c in lv:
            v = to_categorical(c, size)
            bv = bv + v

        labelsarray[index, :] = bv
        index += 1
    return labelsarray


def learning_rate_schedule(model, epoch, nb_epochs):
    # TODO Pass this as an argument
    if epoch < 0.8 * nb_epochs:
        K.set_value(model.optimizer.lr, 0.001)
    else:
        K.set_value(model.optimizer.lr, 0.0001)
