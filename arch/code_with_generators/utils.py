import smtplib
import csv
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import scipy.io as spio
from collections import Counter
import pickle


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


def convert_inceptionv3(model, tf_weights):
    """
    Converts Inception V3 caffe model to keras. We use this great implementation:
    https://github.com/h-bo/tsn-tensorflow
    To obtain our tensorflow weights.
    """
    # Load tensorflow model for inception v3
    # Go through each layer of keras inception v3 (model) and get the tensorflow weights from the checkpoint file
    pass


 def get_UCF_classes(self):
    """Extract the classes from our data. If we want to limit them,
    only return the classes we need."""
    classes = []
    for item in self.data_list:
        if item[1] not in classes:
            classes.append(item[1])

    # Sort them.
    classes = sorted(classes)

    # Return.
    if self.class_limit is not None:
        return classes[:self.class_limit]
    else:
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

def learning_rate_schedule(model, epoch, nb_epochs):
    if epoch < 0.9 * nb_epochs:
        K.set_value(model.optimizer.lr, 0.001)
    else:
        K.set_value(model.optimizer.lr, 0.0001)
