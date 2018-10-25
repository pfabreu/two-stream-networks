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


def bbs_count_barplot(split):
    bbnums = np.zeros(15)
    nums = {}
    with open("files/AVA_" + split + "_Custom_Corrected.csv") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            video_timestamp = row[0] + "," + row[1]
            nums[video_timestamp] = []
    with open("files/AVA_" + split + "_Custom_Corrected.csv") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            video_timestamp = row[0] + "," + row[1]

            bb = row[2] + "," + row[3] + "," + row[4] + "," + row[5]
            if bb not in nums[video_timestamp]:
                nums[video_timestamp].append(bb)
    for key in nums:
        bbnums[len(nums[key])] += 1
    ax = sns.barplot(x=range(1, 15), y=bbnums[1:])
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='gray', rotation=0, xytext=(0, 4),
                    textcoords='offset points')
    # plt.yticks(ax.get_yticks(), ax.get_yticks() * 100.0)
    plt.title("BB counts (" + split + "): ")
    plt.grid(True)


def bbs_dist_barplot(split):
    bbnums = np.zeros(2)
    nums = {}
    with open("files/AVA_" + split + "_Custom_Corrected.csv") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            video_timestamp = row[0] + "," + row[1]
            nums[video_timestamp] = []
    with open("files/AVA_" + split + "_Custom_Corrected.csv") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            video_timestamp = row[0] + "," + row[1]

            bb = row[2] + "," + row[3] + "," + row[4] + "," + row[5]
            if bb not in nums[video_timestamp]:
                nums[video_timestamp].append(bb)
    all_keys = 0
    all_bbs = 0
    for key in nums:
        all_keys += 1
        all_bbs += len(nums[key])
        if len(nums[key]) < 2:
            bbnums[0] += 1
        else:
            bbnums[1] += 1
    bbnums[0] /= all_keys
    bbnums[1] /= all_keys
    ax = sns.barplot(x=["1", ">1"], y=bbnums, palette='spring')
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=20, color='gray', rotation=0, xytext=(0, 4),
                    textcoords='offset points')
    # plt.yticks(ax.get_yticks(), ax.get_yticks() * 100.0)
    plt.title("BB dist (" + split + ")(%): Avg BB/timestamp=" + str(1.0 * all_bbs / all_keys), fontsize=14)
    ax.tick_params(labelsize=12)
    plt.grid(True)


plt.style.use('seaborn-white')
# Show histograms of class distribution for each split

# Show coincidence matrices one at a time
plt.figure()
for i in range(1, 4):

    plt.subplot(1, 3, i)
    if i == 1:
        bbs_count_barplot('Train')
    if i == 2:
        bbs_count_barplot('Val')
    if i == 3:
        bbs_count_barplot('Test')
plt.figure()
for i in range(1, 4):
    plt.subplot(1, 3, i)
    if i == 1:
        bbs_dist_barplot('Train')
    elif i == 2:
        bbs_dist_barplot('Val')
    elif i == 3:
        bbs_dist_barplot('Test')
plt.show()
