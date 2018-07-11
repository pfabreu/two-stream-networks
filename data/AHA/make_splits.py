import os
import sys
import numpy as np
import scipy
import scipy.io as spio
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy.misc
import skimage
from skimage.transform import resize
from PIL import Image
from numpy import array


def augment_aha():
    # AHA - Augmented Human Activities dataset
    #label_names = ['2-minute-step']
    label_names = ['2-minute-step', '30-second-chair-stand',
                   '8-feet-up-and-go', 'unipedal-stance']
    root_dir = '/media/pedro/actv3/AHA/'

    i_width = 224
    i_height = 224

    mats_vec = []
    labels_vec = []

    for label in label_names:
        f = 0
        print("Processing " + label)
        for file in os.listdir(root_dir + label + "/"):
            # RGB Images data
            if "Body" not in file and file != "images" and file != "images_aug" and ".avi" not in file:
                f += 1
                print "\t" + file
                mats_vec.append(file)
                labels_vec.append(label)
    a = array(mats_vec)
    b = array(labels_vec)
    a = np.stack((a, b), axis=1)
    print a.shape
    trn, val, tst = np.vsplit(a[np.random.permutation(a.shape[0])], (60, 80))
    f = open('splits.txt', 'w')
    for i in range(1, trn.shape[0]):
        f.write(str(trn[i, 0]) + ',' + str(trn[i, 1]) + ',' + 'trn' + '\n')

    for i in range(1, val.shape[0]):
        f.write(str(val[i, 0]) + ',' + str(val[i, 1]) + ',' + 'val' + '\n')

    for i in range(1, tst.shape[0]):
        f.write(str(tst[i, 0]) + ',' + str(tst[i, 1]) + ',' + 'tst' + '\n')

if __name__ == '__main__':
    augment_aha()
