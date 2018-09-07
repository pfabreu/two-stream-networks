import csv
import os
import sys
from distutils.dir_util import copy_tree

gen_type = 'train'
_DATA_DIR = "flow_" + gen_type + "/"
_OUT_DIR = "transferDrive/flow_gray_" + gen_type + "/"
snippets_video = []
snippets_time = []
# snippets_bb = []

with open('AVA2.1/AVA_' + gen_type.title() + '_Custom_Corrected.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])
        # snippets_bb.append(row[2] + "_" + row[3] + "_" + row[4] + "_" + row[5])  # xmin,ymin,xmax,ymax

currentVideo = ''
currentTime = ''
# currentBB = ''

for i in range(len(snippets_video)):
    if i % 100 == 0:
        print("Video #" + str(i) + " of " + str(len(snippets_video)) + ".")
    if currentVideo == snippets_video[i] and currentTime == snippets_time[i]:
        pass
    else:
        currentVideo = snippets_video[i]
        currentTime = snippets_time[i]
        # currentBB = snippets_bb[i]

        if currentTime[0] == "0":
            currentTimeStr = currentTime[1:]
        else:
            currentTimeStr = currentTime  # Fixed this :D

        # x and y folder
        in_folder_x = _DATA_DIR + "x/" + currentVideo + "_" + currentTimeStr
        out_folder_x = _OUT_DIR + "x/" + currentVideo + "_" + currentTimeStr
        in_folder_y = _DATA_DIR + "y/" + currentVideo + "_" + currentTimeStr
        out_folder_y = _OUT_DIR + "y/" + currentVideo + "_" + currentTimeStr

        if not os.path.exists(in_folder_x):
            print(in_folder_x + " not found")
            sys.exit(0)
        else:
            if not os.path.exists(out_folder_x):
                # Copy file over
                copy_tree(in_folder_x, out_folder_x)
        if not os.path.exists(in_folder_y):
            print(in_folder_y + " not found")
            sys.exit(0)
        else:
            if not os.path.exists(out_folder_y):
                # Copy file over
                copy_tree(in_folder_y, out_folder_y)
