import csv
import random


_DATA_DIR = "/media/pedro/actv4/AVA/segments"

set_type = "train"
_DATA_DIR = _DATA_DIR + "_" + set_type + "/"
snippets_video = []
snippets_time = []
snippets_action = []
snippets_x1 = []
snippets_y1 = []
snippets_x2 = []
snippets_y2 = []

if set_type == "validation":
    with open('ava_val_v2.1.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            snippets_video.append(row[0])
            snippets_time.append(row[1])
            snippets_x1.append(row[2])  # x top_left
            snippets_y1.append(row[3])  # y top_left
            snippets_x2.append(row[4])  # x bottom right
            snippets_y2.append(row[5])  # y bottom right
            snippets_action.append(row[6])

elif set_type == "train":
    with open('ava_train_v2.1.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            snippets_video.append(row[0])
            snippets_time.append(row[1])
            snippets_x1.append(row[2])
            snippets_y1.append(row[3])
            snippets_x2.append(row[4])
            snippets_y2.append(row[5])
            snippets_action.append(row[6])

snp_vid = ""
snp_time = 0
snips = []
grouped_actions = []
grouped_bbs = []
snp_vid = snippets_video[0]
snp_time = snippets_time[0]
grouped_actions.append(snp_vid)
grouped_actions.append(snp_time)

for i in range(0, len(snippets_video)):
    if snp_vid == snippets_video[i] and snp_time == snippets_time[i]:
        grouped_actions.append(snippets_action[i])
        grouped_bbs.append((snippets_x1[i], snippets_y1[
                           i], snippets_x2[i], snippets_y2[i]))
    else:
        # print(grouped_actions)
        for bb in grouped_bbs:
            grouped_actions.append(bb)
        snips.append(grouped_actions)
        # snips.append(grouped_bbs)
        grouped_bbs = []
        grouped_actions = []
        grouped_actions.append(snippets_video[i])
        grouped_actions.append(snippets_time[i])
        # TODO ADD THE FIRST ACTION!!
        grouped_actions.append(snippets_action[i])
        grouped_bbs.append((snippets_x1[i], snippets_y1[
                           i], snippets_x2[i], snippets_y2[i]))

    snp_vid = snippets_video[i]
    snp_time = snippets_time[i]

# Remove segments with no labels
somelist = [x for x in snips if not len(x) == 2]

snippets_video = []
snippets_time = []
snippets_action = []
snippets_x1 = []
snippets_y1 = []
snippets_x2 = []
snippets_y2 = []

with open('AVA2.1/ava_mini_split_train.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])
        snippets_x1.append(row[2])
        snippets_y1.append(row[3])
        snippets_x2.append(row[4])
        snippets_y2.append(row[5])
        snippets_action.append(row[6])

snp_vid = ""
snp_time = 0
snips = []
grouped_actions = []
grouped_bbs = []
snp_vid = snippets_video[0]
snp_time = snippets_time[0]
grouped_actions.append(snp_vid)
grouped_actions.append(snp_time)

for i in range(0, len(snippets_video)):
    if snp_vid == snippets_video[i] and snp_time == snippets_time[i]:
        grouped_actions.append(snippets_action[i])
        grouped_bbs.append((snippets_x1[i], snippets_y1[
                           i], snippets_x2[i], snippets_y2[i]))
    else:
        # print(grouped_actions)
        for bb in grouped_bbs:
            grouped_actions.append(bb)
        snips.append(grouped_actions)
        # snips.append(grouped_bbs)
        grouped_bbs = []
        grouped_actions = []
        grouped_actions.append(snippets_video[i])
        grouped_actions.append(snippets_time[i])
        # TODO ADD THE FIRST ACTION!!
        grouped_actions.append(snippets_action[i])
        grouped_bbs.append((snippets_x1[i], snippets_y1[
                           i], snippets_x2[i], snippets_y2[i]))

    snp_vid = snippets_video[i]
    snp_time = snippets_time[i]

# Remove segments with no labels
prev_somelist = [x for x in snips if not len(x) == 2]










# print somelist[1]
linecount = len(somelist)
k = int(len(somelist) * 0.2)
# print random_linenos
with open('AVA2.1/ava_mini_split_ext_' + set_type + '.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(0, k):
        row = somelist[randint(0,linecount)]
        video = row[0]
        keyframe = row[1]
        
        row = somelist[i]
        # Remove zeros from keyframe
        vid_name = video + "_" + keyframe.lstrip("0") + ".avi"
        actions_and_bbs = row[2:]
        # Insert into a file each action + corresponding bb
        number_of_actions = len(actions_and_bbs)
        print actions_and_bbs
        print "LEN: " + str(len(actions_and_bbs))
        for j in range(0, number_of_actions / 2):
            action = actions_and_bbs[j]
            bb = actions_and_bbs[j + number_of_actions / 2]
            print "ACTION: " + str(action) + "   BB:  " + str(bb)
            writer.writerow(
                [video, keyframe, float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]), action])
        print "Done writing a seg. Next seg below: "
