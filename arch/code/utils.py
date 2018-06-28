import smtplib
import csv
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import backend as K
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


def majorityVoting(voting_pose, voting_obj, voting_human):
    # Convert list of list (for obj and human) to list
    voting_obj = [item for sublist in voting_obj for item in sublist]
    voting_human = [item for sublist in voting_human for item in sublist]
    # Pick major vote among the arrays and write to the csv
    action_list = []
    obj_counter = Counter(voting_obj)
    human_counter = Counter(voting_human)
    pose_counter = Counter(voting_pose)
    pose_value, count = pose_counter.most_common()[0]  # Get the single most voted pose
    action_list.append(pose_value)
    o = obj_counter.most_common()[:3]  # If there are less than 3, it's no problem
    for tup in o:
        action_list.append(tup[0])  # tup[0] is value, tup[1] is count
    h = human_counter.most_common()[:3]
    for tup in h:
        action_list.append(tup[0])
    return action_list


def pred2classes(ids, predictions, output_csv):
    pose_list = []
    obj_list = []
    human_list = []

    OBJECT_THRESHOLD = 0.4
    HUMAN_THRESHOLD = 0.4

    i = 0
    for entry in predictions:
        for action_type in entry:
            arr = np.array(action_type)

            if i == 0:
                r = arr.argsort()[-1:][::-1]
                pose_list.append(r[0])
            elif i == 1:
                r = arr.argsort()[-3:][::-1]  # Get the three with the highest probabilities
                # TODO Get top 3 and check if they are above threshold
                p = r.tolist()
                prediction_list = []
                # print( p
                for pred in p:
                    if arr[pred] > OBJECT_THRESHOLD:
                        # print( arr[pred]
                        prediction_list.append(pred)
                # print( prediction_list
                obj_list.append(prediction_list)
            elif i == 2:
                r = arr.argsort()[-3:][::-1]
                # TODO Get top 3 and check if they are above threshold
                p = r.tolist()
                # print( p
                for pred in p:
                    if arr[pred] > HUMAN_THRESHOLD:
                        prediction_list.append(pred)
                        # print( prediction_list
                human_list.append(prediction_list)
        i += 1
    voting_pose = []
    voting_obj = []
    voting_human = []
    i = 0

    with open(output_csv, "a") as output_file:
        for entry in ids:
            idx = entry.split("@")
            f = int(idx[6]) - 1
            voting_pose.append(pose_list[i])  # pose was a 1 element list
            voting_obj.append(obj_list[i])
            voting_human.append(human_list[i])
            if f == 4:
                action_list = majorityVoting(voting_pose, voting_obj, voting_human)
                # Write csv lines
                video_name = idx[0]
                timestamp = idx[1]
                bb_topx = idx[2]
                bb_topy = idx[3]
                bb_botx = idx[4]
                bb_boty = idx[5]

                for action in action_list:
                    line = video_name + "," + timestamp + "," + bb_topx + "," + bb_topy + "," + bb_botx + "," + bb_boty + "," + str(action)
                    output_file.write("%s\n" % line)
                # Reset the voting arrays
                voting_pose = []
                voting_obj = []
                voting_human = []

        i += 1


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


def convert_inceptionv3(model, tf_weights):
    """
    Converts Inception V3 caffe model to keras. We use this great implementation:
    https://github.com/h-bo/tsn-tensorflow
    To obtain our tensorflow weights.
    """
    # Load tensorflow model for inception v3
    # Go through each layer of keras inception v3 (model) and get the tensorflow weights from the checkpoint file
    pass


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
    if epoch < 0.9 * nb_epochs:
        K.set_value(model.optimizer.lr, 0.001)
    else:
        K.set_value(model.optimizer.lr, 0.0001)
