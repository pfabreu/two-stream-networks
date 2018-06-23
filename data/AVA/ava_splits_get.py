import csv
import os
from shutil import copyfile

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
    with open('ava_mini_split_validation.csv') as csvDataFile:
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
    with open('ava_mini_split_train.csv') as csvDataFile:
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
        # Add the first action
        grouped_actions.append(snippets_action[i])
        grouped_bbs.append((snippets_x1[i], snippets_y1[
                           i], snippets_x2[i], snippets_y2[i]))

    snp_vid = snippets_video[i]
    snp_time = snippets_time[i]

# Remove segments with no labels/actions (i.e len = 2)
somelist = [x for x in snips if not len(x) == 2]

COPY_DIR = "/media/pedro/actv4/AVA-split/segments_" + set_type + "/"
print len(somelist)
for i in range(0, len(somelist)):
    # Get video name
    row = somelist[i]
    vidname = row[0] + "_" + row[1].lstrip("0") + ".avi"
    # Check if vidname exists
    # If it does copy it to COPY_DIR + vidname
    # Else break
    src = _DATA_DIR + vidname
    if os.path.exists(src):
        dest = COPY_DIR + vidname
        if not os.path.exists(dest):
            print str(i) + "/" + str(len(somelist)) + " Copying " + vidname
            copyfile(src, dest)
    else:
        print vidname
        print "Aborting, original snippet not found"
        break

print "Finished copying segments"
