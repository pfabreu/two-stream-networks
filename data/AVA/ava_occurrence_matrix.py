import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import nan

plt.style.use('seaborn-white')
# Show histograms of class distribution for each split


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


def coincidence_matrix(file):
    cm = np.load(file)
    cm = cm.astype(int)
    print(cm.shape)
    return cm

plt.figure()
# Show coincidence matrices one at a time
normalize = True
classes = get_AVA_classes('files/ava_action_list_custom.csv')
classes['label_name'].insert(0, 'none')
cm = coincidence_matrix("occurrenceMatrices/TrainLabelsConMatrix.npy")
colormap = sns.cubehelix_palette(8, start=.5, rot=-.75)
if normalize is True:
    row_sums = cm.sum(axis=1)
    cma = 1.0 * cm / row_sums[:, np.newaxis]
    cma = np.round(cma, decimals=2) + np.zeros(cm.shape)
    cma = np.nan_to_num(cma)
    g = sns.heatmap(cma, xticklabels=classes['label_name'], yticklabels=classes['label_name'], annot=True, fmt=".2f", cmap=colormap, linewidths=0.5, linecolor='gray', cbar=True)
else:
    cma = cm
    g = sns.heatmap(cma, xticklabels=classes['label_name'], yticklabels=classes['label_name'], annot=True, fmt="d", cmap=colormap, linewidths=0.5, linecolor='gray', cbar=True)
plt.title("What actions co-occur the most? (Train), row-wise")
print(cma)
i = 0
for ytick_label, xtick_label in zip(g.axes.get_yticklabels(), g.axes.get_xticklabels()):
    if i != 0:
        idx = i - 1

    if i <= 10:
        ytick_label.set_color("r")
        xtick_label.set_color("r")

    elif i <= 22:
        ytick_label.set_color("b")
        xtick_label.set_color("b")
    else:
        ytick_label.set_color("g")
        xtick_label.set_color("g")
    i += 1
plt.show()

cm = coincidence_matrix("occurrenceMatrices/ValidationLabelsConMatrix.npy")
if normalize is True:
    row_sums = cm.sum(axis=1)
    cma = 1.0 * cm / row_sums[:, np.newaxis]
    cma = np.round(cma, decimals=2) + np.zeros(cm.shape)
    cma = np.nan_to_num(cma)
    g = sns.heatmap(cma, xticklabels=classes['label_name'], yticklabels=classes['label_name'], annot=True, fmt=".2f", cmap=colormap, linewidths=0.5, linecolor='gray', cbar=True)
else:
    cma = cm
    g = sns.heatmap(cma, xticklabels=classes['label_name'], yticklabels=classes['label_name'], annot=True, fmt="d", cmap=colormap, linewidths=0.5, linecolor='gray', cbar=True)
plt.title("What actions co-occur the most? (Validation), row-wise")

i = 0
for ytick_label, xtick_label in zip(g.axes.get_yticklabels(), g.axes.get_xticklabels()):

    if i <= 10:
        ytick_label.set_color("r")
        xtick_label.set_color("r")

    elif i <= 22:
        ytick_label.set_color("b")
        xtick_label.set_color("b")
    else:
        ytick_label.set_color("g")
        xtick_label.set_color("g")
    i += 1
plt.show()

cm = coincidence_matrix("occurrenceMatrices/TestLabelsConMatrix.npy")
cm = cm[1:, 1:]
classes = get_AVA_classes('files/ava_action_list_custom.csv')
if normalize is True:
    row_sums = cm.sum(axis=1)
    cma = 1.0 * cm / row_sums[:, np.newaxis]
    cma = np.round(cma, decimals=2) + np.zeros(cm.shape)
    cma = np.nan_to_num(cma)
    g = sns.heatmap(cma, xticklabels=classes['label_name'], yticklabels=classes['label_name'], annot=True, fmt=".2f", cmap=colormap, linewidths=0.5, linecolor='gray', cbar=True)
else:
    cma = cm
    g = sns.heatmap(cma, xticklabels=classes['label_name'], yticklabels=classes['label_name'], annot=True, fmt="d", cmap=colormap, linewidths=0.5, linecolor='gray', cbar=True)
plt.title("What actions co-occur the most? (Test), row-wise")

i = 0
for ytick_label, xtick_label in zip(g.axes.get_yticklabels(), g.axes.get_xticklabels()):
    if i <= 10:
        ytick_label.set_color("r")
        xtick_label.set_color("r")

    elif i <= 22:
        ytick_label.set_color("b")
        xtick_label.set_color("b")
    else:
        ytick_label.set_color("g")
        xtick_label.set_color("g")
    i += 1
plt.show()
