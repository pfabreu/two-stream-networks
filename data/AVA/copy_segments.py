import csv
import os
import sys
from shutil import copyfile

gen_type = 'test'
_DATA_DIR = "/media/pedro/actv4/ava-split/split_segments_" + gen_type + "/"
_OUT_DIR = "/media/pedro/actv-ssd/segments_" + gen_type + "/"
snippets_video = []
snippets_time = []


with open('files/AVA_' + gen_type.title() + '_Custom_Corrected.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])

currentVideo = ''
currentTime = ''

for i in range(len(snippets_video)):
    if i % 100 == 0:
        print("Video #" + str(i) + " of " + str(len(snippets_video)) + ".")
    if currentVideo == snippets_video[i] and currentTime == snippets_time[i]:
        pass
    else:
        currentVideo = snippets_video[i]
        currentTime = snippets_time[i]

        if currentTime[0] == "0":
            currentTimeStr = currentTime[1:]
        else:
            currentTimeStr = currentTime  # Fixed this :D
        output_folder = currentVideo + "_" + currentTimeStr + ".avi"
        videoPath = _DATA_DIR + currentVideo + "_" + currentTimeStr + ".avi"
        if not os.path.exists(videoPath):
            print(videoPath + " not found")
            sys.exit(0)
        else:
            if not os.path.exists(_OUT_DIR + output_folder):
                # Copy file over
                copyfile(videoPath, _OUT_DIR + output_folder)
