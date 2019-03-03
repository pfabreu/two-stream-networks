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

def learning_rate_schedule(model, epoch, nb_epochs):
    # TODO Pass this as an argument
    if epoch < 0.8 * nb_epochs:
        K.set_value(model.optimizer.lr, 0.001)
    else:
        K.set_value(model.optimizer.lr, 0.0001)

def make_chunks(original_list, size, chunk_size):
    seq = original_list[:size]
    splits = [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]
    return splits


def to_binary_vector(list_classes, size, labeltype):
    """
    Converts list_classes list to binary vector with given size
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
