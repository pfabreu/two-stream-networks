#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:03:14 2018

@author: jantunes
"""

# TESTING SPLIT COMES FROM THE END OF THE TRAINING SPLIT
# TRAINING AND VAL ARE JUST SUBSETS OF MINI_SPLIT_X_BIG
import csv
import numpy as np


def writeCSV(filename, videoName, keyFrame, bb, action):
    csvLine = videoName + ',' + keyFrame + ',' + str(bb[0]) + ',' + str(bb[1]) + ',' + str(bb[2]) + ',' + str(bb[3]) + ',' + str(action) + '\n'
    with open(filename, 'a') as f:
        f.write(csvLine)
    return


TRAINSPLIT = "AVA_Train_Custom.csv"
VALSPLIT = "AVA_Val_Custom.csv"
TESTSPLIT = "AVA_Test_Custom.csv"
NEWTRAINSPLIT = "AVA_Train_Custom_Corrected.csv"
NEWVALSPLIT = "AVA_Val_Custom_Corrected.csv"
NEWTESTSPLIT = "AVA_Test_Custom_Corrected.csv"
FINALNUMBEROFLINES = 2**14  # 16384
snippets_video = []
snippets_time = []
snippets_bb = []
snippets_actions = []
with open(TESTSPLIT) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])
        snippets_bb.append(np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])]))  # xmin,ymin,xmax,ymax
        snippets_actions.append(row[6])

snippets_actions = snippets_actions[1:FINALNUMBEROFLINES]
actions = np.array([int(i) for i in snippets_actions])
hist, b = np.histogram(actions, 80)


trueClasses = [index+1 for index, value in enumerate(hist) if value >= 15]
removedClasses = [index+1 for index, value in enumerate(hist) if value < 15]
removedPoses = [i for i in removedClasses if i < 14]
#poseClasses = np.arange(1,15)
# trueClasses = list(np.union1d(poseClasses,trueClasses)) #Never remove a pose class!
newClasses = np.arange(1, len(trueClasses)+1)
# Now generate CSV's without the forbidden Classes
# To get corrected class do trueClasses.index(oldClass) + 1

# CORRECT TRAINING
snippets_video = []
snippets_time = []
snippets_bb = []
snippets_actions = []

with open(TRAINSPLIT) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])
        snippets_bb.append(np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])]))  # xmin,ymin,xmax,ymax
        snippets_actions.append(row[6])

snippets_video = snippets_video[1:FINALNUMBEROFLINES]
snippets_time = snippets_time[1:FINALNUMBEROFLINES]
snippets_bb = snippets_bb[1:FINALNUMBEROFLINES]
snippets_actions = snippets_actions[1:FINALNUMBEROFLINES]

i = 0
while(i < len(snippets_actions)):
    action = int(snippets_actions[i])
    if action in trueClasses:
        newAction = trueClasses.index(action) + 1
        writeCSV(NEWTRAINSPLIT, snippets_video[i], snippets_time[i], snippets_bb[i], newAction)
    elif action in removedPoses:
        # Remove all actions corresponding to this bb
        currentbb = snippets_bb[i].tolist()
        while(currentbb == snippets_bb[i].tolist()):
            i += 1
        i -= 1
    i += 1

# CORRECT VAL
snippets_video = []
snippets_time = []
snippets_bb = []
snippets_actions = []

with open(VALSPLIT) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])
        snippets_bb.append(np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])]))  # xmin,ymin,xmax,ymax
        snippets_actions.append(row[6])

snippets_video = snippets_video[1:FINALNUMBEROFLINES]
snippets_time = snippets_time[1:FINALNUMBEROFLINES]
snippets_bb = snippets_bb[1:FINALNUMBEROFLINES]
snippets_actions = snippets_actions[1:FINALNUMBEROFLINES]

i = 0
while(i < len(snippets_actions)):
    action = int(snippets_actions[i])
    if action in trueClasses:
        newAction = trueClasses.index(action) + 1
        writeCSV(NEWVALSPLIT, snippets_video[i], snippets_time[i], snippets_bb[i], newAction)
    elif action in removedPoses:
        # Remove all actions corresponding to this bb
        currentbb = snippets_bb[i].tolist()
        while(currentbb == snippets_bb[i].tolist()):
            i += 1
        i -= 1
    i += 1
# CORRECT TEST
snippets_video = []
snippets_time = []
snippets_bb = []
snippets_actions = []

with open(TESTSPLIT) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])
        snippets_bb.append(np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])]))  # xmin,ymin,xmax,ymax
        snippets_actions.append(row[6])

snippets_video = snippets_video[1:FINALNUMBEROFLINES]
snippets_time = snippets_time[1:FINALNUMBEROFLINES]
snippets_bb = snippets_bb[1:FINALNUMBEROFLINES]
snippets_actions = snippets_actions[1:FINALNUMBEROFLINES]

i = 0
while(i < len(snippets_actions)):
    action = int(snippets_actions[i])
    if action in trueClasses:
        newAction = trueClasses.index(action) + 1
        writeCSV(NEWTESTSPLIT, snippets_video[i], snippets_time[i], snippets_bb[i], newAction)
    elif action in removedPoses:
        # Remove all actions corresponding to this bb
        currentbb = snippets_bb[i].tolist()
        while(currentbb == snippets_bb[i].tolist()):
            i += 1
        i -= 1
    i += 1
