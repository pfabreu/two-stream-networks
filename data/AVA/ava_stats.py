import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


def split_barplot(split):
    types = [0, 0, 0]
    names = ["pose", "obj", "human"]
    with open("files/AVA_" + split + "_Custom_Corrected.csv") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            tp = int(row[6])
            if tp <= 10:
                types[0] += 1
            elif tp <= 22:
                types[1] += 1
            else:
                types[2] += 1
    types = [100.0 * float(i) / sum(types) for i in types]
    ax = sns.barplot(x=names, y=types, palette=['red', 'green', 'blue'])
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='gray', rotation=0, xytext=(0, 4),
                    textcoords='offset points')
    # plt.yticks(ax.get_yticks(), ax.get_yticks() * 100.0)
    plt.title("Type (" + split + ")(%)")
    plt.grid(True)
    t = 0
    plt.xticks(range(0, 3))
    for xtick_label in ax.axes.get_xticklabels():
        if t == 0:
            xtick_label.set_color("r")
        elif t == 1:
            xtick_label.set_color("g")
        else:
            xtick_label.set_color("b")
        t += 1


def dist_barplot(split):
    classes = []
    with open("files/AVA_" + split + "_Custom_Corrected.csv") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            classes.append(int(row[6]))
    ax = sns.distplot(classes, color="y", bins=range(1, 31))
    plt.xlim(0, 30)
    plt.ylim(0, 0.2)
    # plt.yticks(ax.get_yticks(), ax.get_yticks() * 100.0)
    plt.title("Class probability distribution (" + split + ")(%)")
    plt.grid(True)
    t = 0
    plt.xticks(range(0, 31))
    for xtick_label in ax.axes.get_xticklabels():
        if t <= 10:
            xtick_label.set_color("r")
        elif t <= 22:
            xtick_label.set_color("b")
        else:
            xtick_label.set_color("g")
        t += 1


def count_barplot(split):
    types = np.zeros(30)
    cls = get_AVA_classes('files/ava_action_list_custom.csv')
    classes = range(30)
    with open("files/AVA_" + split + "_Custom_Corrected.csv") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            tp = int(row[6]) - 1
            types[tp] += 1
    types = list(types)
    colors = []
    for t in range(30):
        if t <= 10:
            colors.append("r")
        elif t <= 22:
            colors.append("b")
        else:
            colors.append("g")
    trips = zip(types, classes, colors)
    # print(trips)
    trips.sort(reverse=True)
    # print(trips)
    types = [x[0] for x in trips]
    classes = [cls['label_name'][x[1]] for x in trips]
    colors = [x[2] for x in trips]
    plt.xticks(rotation=90)

    ax = sns.barplot(x=classes, y=types, palette=colors)
    plt.grid(True)
    plt.title("Class sample histogram (ordered) (" + split + ")")
    t = 0
    for xtick_label in ax.axes.get_xticklabels():
        xtick_label.set_color(colors[t])
        t += 1


plt.style.use('seaborn-white')
# Show histograms of class distribution for each split

# Show coincidence matrices one at a time
plt.figure()
for i in range(1, 4):

    plt.subplot(1, 3, i)
    if i == 1:
        split_barplot('Train')
    if i == 2:
        split_barplot('Val')
    if i == 3:
        split_barplot('Test')

plt.figure()
for i in range(1, 4):
    plt.subplot(1, 3, i)
    if i == 1:
        dist_barplot('Train')
    elif i == 2:
        dist_barplot('Val')
    elif i == 3:
        dist_barplot('Test')

plt.figure()
for i in range(1, 4):
    plt.subplot(1, 3, i)
    if i == 1:
        count_barplot('Train')
    elif i == 2:
        count_barplot('Val')
    elif i == 3:
        count_barplot('Test')


plt.show()
