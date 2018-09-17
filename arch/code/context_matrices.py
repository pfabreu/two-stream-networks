import csv
import numpy as np

INPATHS = ["AVA_train_Custom_Corrected.csv", "AVA_validation_Custom_Corrected.csv", "AVA_Test_Custom_Corrected.csv"]
OUTPATHS = ["TrainLabelsConMatrix.npy", "ValidationLabelsConMatrix.npy", "TestLabelsConMatrix.npy"]
NCLASSES = 30
snippets_video = []
snippets_time = []
snippets_bb = []
snippets_actions = []


def getRelevantIndexes(videoID, keyFrame, bb, snippets_video, snippets_time, snippets_bb):
    indexes = []

    IDindices = [i for i, x in enumerate(snippets_video) if x == videoID]

    keyFrameIndices = [i for i, x in enumerate(snippets_time) if int(x) == int(keyFrame)]
    #bbIndices = [i for i, x in enumerate(snippets_bb) if sum(currentBB == snippets_bb[i]) == len(currentBB)]

    indexes = set(IDindices) - (set(IDindices) - set(keyFrameIndices))

    #indexes = set(IDindices) & set(keyFrameIndices) & set (bbIndices);
    return list(indexes)


def getActions(indexes, currentBB, snippets_actions, snippets_bb):
    targetActions = []
    neighborActions = []
    for i in indexes:
        if (sum(currentBB == snippets_bb[i]) == len(currentBB)):
            # This is an action of our current BB
            targetActions.append(snippets_actions[i])
        else:
            # This is an action of a neighbor
            neighborActions.append(snippets_actions[i])
    return targetActions, neighborActions


for i in range(len(INPATHS)):
    INPATH = INPATHS[i]
    OUTPATH = OUTPATHS[i]
    with open(INPATH) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            snippets_video.append(row[0])
            snippets_time.append(row[1])
            snippets_bb.append(np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])]))  # xmin,ymin,xmax,ymax
            snippets_actions.append(row[6])

    confMatrix = np.zeros([NCLASSES + 1, NCLASSES + 1])  # So I don't have to mess about with indexes. Line 0 and Column 0 do not matter.

    currentVideoID = ''
    currentKeyFrame = -1
    currentBB = []
    for i in range(len(snippets_video)):
        if (snippets_video[i] == currentVideoID) and \
                (snippets_time[i] == currentKeyFrame) and \
                (sum(currentBB == snippets_bb[i]) == len(currentBB)):
            # This is the same ID,Keyframe,BB as last time. skip this one.
            pass
        else:
            # It's a new BB, update current values
            currentVideoID = snippets_video[i]
            currentKeyFrame = snippets_time[i]
            currentBB = snippets_bb[i]
            # Find the indexes of all actions in the current frame
            indexes = getRelevantIndexes(currentVideoID, currentKeyFrame, currentBB, snippets_video, snippets_time, snippets_bb)
            # Get their labels
            # Indexes contains actions of the target and the neighbors. Separate them using the currentBB check!
            targetActions, neighborActions = getActions(indexes, currentBB, snippets_actions, snippets_bb)
            # Add these labels to the Conf Matrix
            # Create the line vector with the labels
            currentLine = np.zeros([NCLASSES + 1])
            for neighborLabel in neighborActions:
                currentLine[int(neighborLabel)] += 1

            for targetLabel in targetActions:
                # Add it on line# Label
                confMatrix[int(targetLabel), :] += currentLine
    np.save(OUTPATH, confMatrix)
