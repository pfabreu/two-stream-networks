import numpy as np
import utils
from rgb_data import get_AVA_set
import matplotlib.pyplot as plt
import seaborn as sns
import voting
import pickle
import sys
import csv


# prior distributions are commonly used as prior distributions in Bayesian statistics, and in fact the prior distribution
# is the conjugate prior of the categorical distribution and multinomial distribution.

# File with predictions
stream = "rgb"
filter_type = "gauss"
filename = "thresholds/" + stream + "_" + filter_type + "/predictions_" + stream + "_" + filter_type + "_1807241628.pickle"
with open(filename, 'rb') as handle:
    predictions = pickle.load(handle)


root_dir = '../../data/AVA/files/'

# Load groundtruth (test or val set)
# Load list of action classes and separate them (from utils_stream)
classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')
print(classes)
f = np.zeros(30)
with open("../../data/AVA/files/AVA_Test_Custom_Corrected.csv") as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        f[int(row[6]) - 1] += 1
print(f)
p = np.divide(f, np.sum(f))
print(p)
a = np.zeros(30)

partition = get_AVA_set(classes=classes, filename=root_dir + "AVA_Test_Custom_Corrected.csv", soft_sigmoid=True)
test_splits = utils.make_chunks(original_list=partition, size=len(partition), chunk_size=2**11)
# First "test" where all a's are still 0
# Pick a's: Select random combinations from a set of values for each class, making sure they aren't repeated?
# Just test uniform value for now (all a_L's the same)
a_set = [0, 10, 100, 1000, 5000, 10000]
i = 1
bls = []
for a_s in a_set:
    plt.subplot(3, 2, i)
    i += 1
    x_axis = []
    colors = []
    for m in classes['label_id']:
        ms = int(m)
        if ms <= 10:
            colors.append('red')
        elif ms < 22:
            colors.append('blue')
        else:
            colors.append('green')
    for m in classes['label_name']:
        ms = m.split("/")[-1]
        x_axis.append(ms)

    b = np.zeros(30)
    for bs in range(len(b)):
        b[bs] = (f[bs] + a_s) / np.sum(f + a_s)
    sbs = np.sum(b)
    for bs in range(len(b)):
        b[bs] /= sbs
    print(np.sum(b))
    g = sns.barplot(x=x_axis, y=b)
    plt.xticks(rotation=-90)
    plt.title("Thresholding Prior, all a_L's = " + str(a_s))
    plt.grid(True)
    t = 0
    for xtick_label in g.axes.get_xticklabels():
        if t <= 9:
            xtick_label.set_color("r")
        elif t < 22:
            xtick_label.set_color("b")
        else:
            xtick_label.set_color("g")
        t += 1
    b = np.split(b, [10, 22])  # Has to be done due to shape of predictions .pkl file
    bls.append(b)


plt.show()

# Save combo files

# Voting (will output a lot of csvs)
a_idx = 0
for b in bls:

    print("Regularizing predictions from file: " + filename)
    with open(filename, 'rb') as handle:
        predictions = pickle.load(handle)

    for tc in range(len(predictions)):
        for predtype in range(3):
            print(len(predictions[tc][predtype]))
            # print("Pre-reg:")
            print(predictions[tc][predtype])
            # NOTE Element wise regulation by the prior prior
            print("Prior:")
            predictions[tc][predtype] = predictions[tc][predtype] * b[predtype]
            print(predictions[tc][predtype])
            # Regularize predictions given the combination

    pose_votes = {}
    obj_votes = {}
    human_votes = {}

    for row in partition:
        row = row.split("@")
        i = row[0] + "@" + row[1] + "@" + str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
        pose_votes[i] = np.zeros(utils.POSE_CLASSES)
        obj_votes[i] = np.zeros(utils.OBJ_HUMAN_CLASSES)
        human_votes[i] = np.zeros(utils.HUMAN_HUMAN_CLASSES)

    pred_chunk = 0
    for testIDS in test_splits:
        voting.pred2classes(testIDS, predictions[pred_chunk], pose_votes, obj_votes, human_votes, thresh=0.2)
        pred_chunk += 1

    result_csv = filename.split(".")[0] + "_" + str(a_set[a_idx]) + ".csv"
    print(result_csv)
    result_csv = result_csv.split("/")[1] + "/" + result_csv.split("/")[2]

    result_csv = "prior/" + result_csv
    a_idx += 1
    print(result_csv)
    with open(result_csv, "a") as output_file:
        for key in pose_votes:
            idx = key.split("@")
            actions = []
            pv = pose_votes[key]
            pose_vote = pv.argmax(axis=0) + 1
            actions.append(pose_vote)

            # Get 3 top voted object
            ov = obj_votes[key]
            top_three_obj_votes = ov.argsort()[-3:][::-1] + utils.POSE_CLASSES + 1
            for t in top_three_obj_votes:
                if t != 0:  # Often there might only be two top voted or one
                    actions.append(t)
            # Get 3 top voted human
            hv = human_votes[key]
            top_three_human_votes = hv.argsort()[-3:][::-1] + utils.POSE_CLASSES + utils.OBJ_HUMAN_CLASSES + 1
            for t in top_three_human_votes:
                if t != 0:  # Often there might only be two top voted or one
                    actions.append(t)

            video_name = idx[0]
            timestamp = idx[1]
            bb_topx = idx[2]
            bb_topy = idx[3]
            bb_botx = idx[4]
            bb_boty = idx[5]
            for a in actions:
                line = video_name + "," + timestamp + "," + bb_topx + "," + bb_topy + "," + bb_botx + "," + bb_boty + "," + str(a)
                output_file.write("%s\n" % line)
