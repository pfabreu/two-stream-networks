"""
Generates a context file from the output predictions of a 2 stream model
"""

import csv
import numpy as np
from scipy.spatial import distance
import stream.utils as utils
from keras.utils import to_categorical

snippets_video = []
snippets_time = []
snippets_bb = []
snippets_actions = []
VALCSVPATH = "output_val_1806071730.csv"
OUTCSVFILENAME = "starter_list.csv"

NUMBEROFNEIGHBORS = 3
NCLASSES = utils.POSE_CLASSES + utils.OBJ_HUMAN_CLASSES + utils.HUMAN_HUMAN_CLASSES
LOOKBACK = 1
LOOKFORWARD = 1


def writeCSV(filename, XLine, videoName, keyFrame, bb):
    csvLine = videoName + ',' + keyFrame + ',' + str(bb[0]) + ',' + str(bb[1]) + ',' + str(bb[2]) + ',' + str(bb[3]) + ',' + str(XLine) + '\n'
    with open(filename, 'a') as f:
        f.write(csvLine)
    return


def createLabelVector(NUMBEROFNEIGHBORS, NCLASSES, distances, actions_dict, TIMEFRAME):
    currentActions = np.zeros([NUMBEROFNEIGHBORS*NCLASSES])
    if(TIMEFRAME == "present"):
        for select in range(len(distances)):
            if(select < NUMBEROFNEIGHBORS):
                currentKey = distances[select][0]
                listClasses = actions_dict[currentKey]
                currentLabelVector = np.zeros(NCLASSES)
                for l in listClasses:
                    v = to_categorical(l, NCLASSES)
                    currentLabelVector = currentLabelVector + v
                currentActions[select*NCLASSES:(select+1)*NCLASSES] = currentLabelVector
    elif(TIMEFRAME == "past"):
        for select in range(1, len(distances)):
            if(select < NUMBEROFNEIGHBORS):
                currentKey = distances[select][0]
                listClasses = actions_dict[currentKey]
                currentLabelVector = np.zeros(NCLASSES)
                for l in listClasses:
                    v = to_categorical(l, NCLASSES)
                    currentLabelVector = currentLabelVector + v
                currentActions[select*NCLASSES:(select+1)*NCLASSES] = currentLabelVector

    return currentActions


def calcDistances(actions_dict, current_bb):
    centerX = (current_bb[0] + (current_bb[2]-current_bb[0])/2.0)
    centerY = (current_bb[1] + (current_bb[3]-current_bb[1])/2.0)
    center = (centerX, centerY)
    distances = []
    for key in actions_dict:
        keysplit = key.split()
        floatkey = np.zeros(4)
        floatkey[0] = float(keysplit[0][1:])
        floatkey[1] = float(keysplit[1])
        floatkey[2] = float(keysplit[2])
        if keysplit[3][-1] == ']':
            floatkey[3] = float(keysplit[3][:-1])
        else:
            floatkey[3] = float(keysplit[3])
        currentCenterX = floatkey[0] + (floatkey[2]-floatkey[0])/2.0
        currentCenterY = floatkey[1] + (floatkey[3]-floatkey[1])/2.0
        currentCenter = (currentCenterX, currentCenterY)
        dist = distance.euclidean(center, currentCenter)
        distances.append((key, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances


def getActionDict(indexes, snippets_bb, snippets_actions, current_bb):
    actions_dict = {}

    for i in indexes:
        if not (sum(currentBB == snippets_bb[i]) == len(currentBB)):
            key = str(snippets_bb[i])
            if key in actions_dict:
                actions_dict[key].append(int(snippets_actions[i])-1)
            else:
                actions_dict[key] = []
                actions_dict[key].append(int(snippets_actions[i])-1)
    return actions_dict


def getRelevantIndexes(videoID, keyFrame, snippets_video, snippets_time):
    indexes = []

    IDindices = [i for i, x in enumerate(snippets_video) if x == videoID]

    keyFrameIndices = [i for i, x in enumerate(snippets_time) if int(x) == int(keyFrame)]
    indexes = set(IDindices) - (set(IDindices) - set(keyFrameIndices))
    return list(indexes)


with open(VALCSVPATH) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])
        snippets_bb.append(np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])]))  # xmin,ymin,xmax,ymax
        snippets_actions.append(row[6])


currentVideoID = ''
currentKeyFrame = ''
currentBB = []
for i in range(len(snippets_video)):
    if i % 500 == 0:
        print("On snippet #" + str(i) + " of " + str(len(snippets_video)) + ".\n")
    if (snippets_video[i] == currentVideoID) and \
        (snippets_time[i] == currentKeyFrame) and \
            (sum(currentBB == snippets_bb[i]) == len(currentBB)):
        # This is the same ID,Keyframe,BB as last time. skip this one.
        pass
    else:
        # This is a different boundingbox
        # Generate an XLine

        currentVideoID = snippets_video[i]
        currentKeyFrame = snippets_time[i]
        currentBB = snippets_bb[i]
        currentIndexes = getRelevantIndexes(currentVideoID, currentKeyFrame, snippets_video, snippets_time)
#        print("ID: " + currentVideoID + ", KF:" + currentKeyFrame + ", BB:")
#        print(currentBB)
#        print("\n")
        actions_dict = getActionDict(currentIndexes, snippets_bb, snippets_actions, currentBB)
        old_actions_dict_list = []
        for timestep in range(1, LOOKBACK+1):
            oldIndexes = []
            oldIndexes = getRelevantIndexes(currentVideoID, int(currentKeyFrame) - timestep, snippets_video, snippets_time)  # Check for behaviour if the videoID/Keyframe combo doesn't exist!
            old_actions_dict_list.append(getActionDict(oldIndexes, snippets_bb, snippets_actions, currentBB))
        currentDistances = calcDistances(actions_dict, currentBB)
        oldDistances = []
        for old_actions_dict in old_actions_dict_list:
            oldDistances.append(calcDistances(old_actions_dict, currentBB))
        # Generate Vector with the data
        presentX = createLabelVector(NUMBEROFNEIGHBORS, NCLASSES, currentDistances, actions_dict, "present")
        oldX = np.zeros([NUMBEROFNEIGHBORS*NCLASSES*LOOKBACK])
        for oldIndex in range(len(old_actions_dict_list)):
            oldX[(oldIndex)*NCLASSES*NUMBEROFNEIGHBORS:(oldIndex+1)*NCLASSES*NUMBEROFNEIGHBORS] = createLabelVector(NUMBEROFNEIGHBORS, NCLASSES, oldDistances[oldIndex], old_actions_dict_list[oldIndex], "past")

        future_actions_dict_list = []
        for timestep in range(1, LOOKFORWARD+1):
            futureIndexes = []
            futureIndexes = getRelevantIndexes(currentVideoID, int(currentKeyFrame) + timestep, snippets_video, snippets_time)  # Check for behaviour if the videoID/Keyframe combo doesn't exist!
            future_actions_dict_list.append(getActionDict(futureIndexes, snippets_bb, snippets_actions, currentBB))
        futureDistances = []
        for future_actions_dict in future_actions_dict_list:
            futureDistances.append(calcDistances(future_actions_dict, currentBB))
        # Generate Vector with the data
        presentX = createLabelVector(NUMBEROFNEIGHBORS, NCLASSES, currentDistances, actions_dict, "present")

        oldX = np.zeros([NUMBEROFNEIGHBORS*NCLASSES*LOOKBACK])
        for oldIndex in range(len(old_actions_dict_list)):
            oldX[(oldIndex)*NCLASSES*NUMBEROFNEIGHBORS:(oldIndex+1)*NCLASSES*NUMBEROFNEIGHBORS] = createLabelVector(NUMBEROFNEIGHBORS, NCLASSES, oldDistances[oldIndex], old_actions_dict_list[oldIndex], "past")

        futureX = np.zeros([NUMBEROFNEIGHBORS*NCLASSES*LOOKFORWARD])
        for futureIndex in range(len(future_actions_dict_list)):
            futureX[(futureIndex)*NCLASSES*NUMBEROFNEIGHBORS:(futureIndex+1)*NCLASSES*NUMBEROFNEIGHBORS] = createLabelVector(NUMBEROFNEIGHBORS, NCLASSES, futureDistances[futureIndex], future_actions_dict_list[futureIndex], "past")

        currentXLine = np.zeros([NUMBEROFNEIGHBORS*NCLASSES*(LOOKBACK+LOOKFORWARD+1)])
        currentXLine[:NUMBEROFNEIGHBORS*NCLASSES] = presentX
        currentXLine[NCLASSES*NUMBEROFNEIGHBORS:NCLASSES*NUMBEROFNEIGHBORS+NUMBEROFNEIGHBORS*NCLASSES*LOOKBACK] = oldX
        currentXLine[NCLASSES*NUMBEROFNEIGHBORS+NUMBEROFNEIGHBORS*NCLASSES*LOOKBACK:] = futureX

        currentXLine = np.array2string(currentXLine, separator=' ', max_line_width=100000)  # hehe
        currentXLine = currentXLine[1:-1]  # Remove trailling "[]"
        writeCSV(OUTCSVFILENAME, currentXLine, currentVideoID, currentKeyFrame, currentBB)
