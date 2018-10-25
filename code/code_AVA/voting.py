import smtplib
import csv
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import scipy.io as spio
from collections import Counter
import pickle
import utils
import sys


def pred2classes(ids, predictions, pose_votes, obj_votes, human_votes, thresh=0.4):

    OBJECT_THRESHOLD = thresh
    HUMAN_THRESHOLD = thresh

    for type_counter in range(3):
        # print("Getting votes for " + str(type_counter))
        for ID, pred in zip(ids, predictions[type_counter]):
            row = ID.split("@")
            i = row[0] + "@" + row[1] + "@" + row[2] + "@" + row[3] + "@" + row[4] + "@" + row[5]

            if type_counter == 0:
                # Each frame gets one vote for pose
                pose_votes[i][pred.argmax(axis=0)] += 1

            elif type_counter == 1:
                # print(pred)
                top_three_idxs = pred.argsort()[-3:][::-1]  # Get the three with the highest probabilities
                # print(top_three_idxs)
                for idx in top_three_idxs:
                    # print(idx)
                    if pred[idx] > OBJECT_THRESHOLD:
                        obj_votes[i][idx] += 1

            elif type_counter == 2:
                top_three_idxs = pred.argsort()[-3:][::-1]  # Get the three with the highest probabilities
                for idx in top_three_idxs:
                    if pred[idx] > HUMAN_THRESHOLD:
                        human_votes[i][idx] += 1
