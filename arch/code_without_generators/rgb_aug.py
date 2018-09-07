import csv
import time
import timeit
import math

import utils
import numpy as np


def main():
    # root_dir = '../../../AVA2.1/' # root_dir for the files
    root_dir = '/home/pedro/Downloads/'

    # Load list of action classes and separate them
    classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')

    # Parameters for training
    params = {'dim': (224, 224), 'batch_size': 32,
              'n_classes': len(classes['label_id']), 'n_channels': 3,
              'shuffle': False, 'nb_epochs': 200, 'model': 'inceptionv3', 'email': True,
              'freeze_all': True, 'conv_fusion': False, 'train_chunk_size': 2**12,
              'validation_chunk_size': 2**12}
    soft_sigmoid = True
    minValLoss = 9999990.0
    print(classes)
    types = np.zeros(len(classes['label_id']))
    avg_samples = 0
    with open(root_dir + "AVA_Train_Custom_Corrected.csv") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            tp = int(row[6]) - 1
            types[tp] += 1
            avg_samples += 1
    avg_samples /= len(classes['label_id'])

    print(types)
    print(avg_samples)
    fds = 0
    for i in range(len(types)):
        if types[i] < avg_samples and types[i] != 0:
            print("Class: " + str(i + 1))
            print("Samples: " + str(types[i]))
            print("Reps: " + str(math.ceil(avg_samples / types[i])))
            fds += 1
if __name__ == '__main__':
    main()
