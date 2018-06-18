import csv
import os
import sys
from sendemail import sendemail
from shutil import copyfile

_DATA_DIR = "segments_validation/"
_OUT_DIR = "/media/jantunes/actv4/ava-split/segments_val/"
snippets_video = []
snippets_time = []


with open('AVA2.1/ava_mini_split_val_big.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])

#Find unique segments
        
currentVideo = '';
currentTime = '';
for i in range(len(snippets_video)):
    if i%100 == 0:
        print("Video #" + str(i) + " of " +str(len(snippets_video)) + ".")
    if currentVideo == snippets_video[i] and currentTime == snippets_time[i]:
        pass
    else:
        currentVideo = snippets_video[i]
        currentTime = snippets_time[i]
        #print(currentTime)
        
        if currentTime[0] == "0":
            currentTimeStr = currentTime[1:]
        else:
            currentTimeStr = currentTime # Fixed this :D
        output_folder = currentVideo + "_" + currentTimeStr + ".avi"
        videoPath = _DATA_DIR + currentVideo + "_" + currentTimeStr + ".avi"
        if not os.path.exists(videoPath):
            print(videoPath + " not found")
            sys.exit(0)
        else:
            if not os.path.exists(_OUT_DIR + output_folder):
                # Copy file over
                copyfile(videoPath, _OUT_DIR + output_folder)
            