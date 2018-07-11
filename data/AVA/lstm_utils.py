#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:30:09 2017

@author: jantunes
"""

import numpy as np
import copy
import math
import sys

def jointLength(currentSkeleton,jointStart,jointEnd):
    #This is working as intended.
    xStart = currentSkeleton[(jointStart)*2]
    yStart = currentSkeleton[(jointStart)*2 +1]
    xEnd   = currentSkeleton[(jointEnd)*2]
    yEnd   = currentSkeleton[(jointEnd)*2+1]
# =============================================================================
#     If at least one of the joints hasn't been detected
#     then I ignore the whole thing
# =============================================================================
    if (xStart != 0) & (xEnd != 0) & (yStart != 0) & (yEnd != 0):
        return math.sqrt((xEnd-xStart)**2 + (yEnd - yStart)**2)
    else:
        return -1

def calculateJointEndCoordinates(currentSkeleton,processedSkeleton,avglength,parentJoint,childJoint):
    # Check if everything is well defined:
    if ((processedSkeleton[parentJoint*2] != sys.float_info.max * -1) 
        and (processedSkeleton[parentJoint*2 + 1] != sys.float_info.max * -1)
        and (currentSkeleton[childJoint*2] != 0)
        and (currentSkeleton[childJoint*2+1] != 0)):
        vector = np.zeros([2,1])
        vector[0] = currentSkeleton[childJoint*2] - currentSkeleton[parentJoint*2]
        vector[1] = currentSkeleton[childJoint*2+1] - currentSkeleton[parentJoint*2+1]
        resizedVector = vector/avglength
        #Add it to the previous joint
        xEnd = processedSkeleton[parentJoint*2] + resizedVector[0] # xCoordinate!
        yEnd = processedSkeleton[parentJoint*2+1] + resizedVector[1] # yCoordinate!
    else:
        xEnd = sys.float_info.max * -1
        yEnd = sys.float_info.max * -1
    return xEnd, yEnd

def processSingleSkeleton(X,dataDimension):
    JOINTS = [[0,1],    #SpineBase to SpineMid
              [1,20],   #SpineMid to SpineShoulder
              [8,9],    #RShoulder to RElbow
              [9,10],   #RElbow to RWrist
              [4,5],    #LShoulder to LElbow
              [5,6],    #LElbow to LWrist
              [16,17],  #RHip to RKnee
              [17,18],  #RKnee to RAnkle
              [12,13],  #LHip to LKnee
              [13,14],  #LKnee to LAnkle
              ];
              #for avg joint length calc
              
    ####### NEED TO REDO FROM HERE IF I WANT TO DO THE PRE PROCESSING ON THIS ONE
    ###
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    currentSkeleton = copy.deepcopy(X)
    #calculate average joint length for normalization
    jointLengthList = [];
    for joint in JOINTS:
        currentLength = jointLength(currentSkeleton,joint[0],joint[1]) 
        if currentLength != -1 :
            jointLengthList.append(currentLength)
    if len(jointLengthList) == 0:
        #I can't calculate ANY JOINT MEASURE. Return -float inf
        processedSkeleton = np.ones([dataDimension,1])
        processedSkeleton = processedSkeleton *-100000
        #print "I just returned a null skeleton cause I couldn't calculate any joint lengths. Something went wrong. please debug!"
        return processedSkeleton

    avglength = np.mean(jointLengthList)
    neckX = currentSkeleton[2]
    neckY = currentSkeleton[3]
    # Center everything around the neck
    for i in range(0,14):
        # For x
        if currentSkeleton[i*2] !=0:
            currentSkeleton[i*2] = currentSkeleton[i*2] - neckX + 10e-15 # to ensure the neck joint isn't 0
        # For y
        if currentSkeleton[i*2+1] !=0:
            currentSkeleton[i*2+1] = currentSkeleton[i*2+1] - neckY + 10e-15 # to ensure the neck joint isn't 0
    
    # Now divide the joints by the average length.
    processedSkeleton = np.zeros([dataDimension,1])
    JOINTPAIRS = [[1,0],    #Neck to Nose
                  [1,2],    #Neck to RShoulder
                  [1,5],    #Neck to LShoulder
                  [1,8],    #Neck to RHip 
                  [1,11],   #Neck to LHip
                  [2,3],    #RShoulder to RElbow
                  [3,4],    #RElbow to RWrist
                  [5,6],    #LShoulder to LElbow
                  [6,7],    #LElbow to LWrist
                  [8,9],    #RHip to RKnee
                  [9,10],   #RKnee to RAnkle
                  [11,12],  #LHip to LKnee
                  [12,13]   #LKnee to LAnkle
                  ]
    # Start with the joints connected to the neck!
    if (currentSkeleton[2] != 0) and (currentSkeleton[3] != 0):
        processedSkeleton[2] = 10e-15
        processedSkeleton[3] = 10e-15
        
        # Process all the joint pairs!
        for joint in JOINTPAIRS:
            parent = joint[0]
            child = joint[1]
            xEnd, yEnd = calculateJointEndCoordinates(currentSkeleton,processedSkeleton,avglength,parent,child)
            processedSkeleton[child*2] = xEnd
            processedSkeleton[child*2+1] = yEnd
    else:
        #If I can't detect the neck I can't process the skeleton.
        processedSkeleton = np.ones([dataDimension,1])
        processedSkeleton = processedSkeleton * sys.float_info.max * -1
        
    return processedSkeleton

def processSkeletonVideo(X,maxFrames,dataDimension):
    video = np.zeros(X.shape)
    for index in range(len(X)):
        currentSkeleton = processSingleSkeleton(X[index][:],dataDimension) 
        video[index,:,None] = currentSkeleton
    return video
    

def findMaximumNumberOfFrames(labelsList):
    maxFrames = 0
    for labelVector in labelsList:
        if (len(labelVector) > maxFrames):
            maxFrames = len(labelVector)
    return maxFrames


def labelDictionary(labelsList):
    uniques = set([]);
    for vector in labelsList:
        current = vector.flatten();
        currentS = set(current);
        for label in currentS:
            uniques.add(label);
    return len(uniques)

def yseg(Y,duration):
    ySeg = np.zeros(Y.shape)
    i = 0
    while i < len(Y):
        if Y[i] >0:
            startingIndex = i;
            while i < len(Y) and Y[i] != 0:
                i = i+1;
            endingIndex = i;
            i = i-1
            #Insert this action in the new vector
            #actionLength = endingIndex - startingIndex + 1
            newStart = startingIndex - duration;
            if(endingIndex + duration < len(Y)):
                newEnd = endingIndex + duration;
            else:
                newEnd = len(Y)
            if(newStart<=0):
                ySeg[0:startingIndex] = 1;
            else:
                ySeg[newStart:startingIndex] = 1;
            ySeg[startingIndex:endingIndex] = 2;
            if(newEnd > endingIndex):
                ySeg[endingIndex:newEnd] = 3;
            else:
                ySeg[endingIndex:len(Y)] = 3;
                
        i = i+1
    return ySeg

def ysegBasic(Y):
    ySeg = np.zeros(Y.shape)
    i = 0
    while i < len(Y):
        if Y[i] >0:
            startingIndex = i;
            while i < len(Y) and Y[i] != 0:
                i = i+1;
            endingIndex = i;
            i = i-1
            #Insert this action in the new vector
            #actionLength = endingIndex - startingIndex + 1
            ySeg[startingIndex:endingIndex] = 1;
        i = i+1
    return ySeg


def prepareBatch(indexes,videosList,labelsList,lookback,overlap,nFeatures,maxLength):
    length = (len(indexes)*maxLength)//(lookback-overlap) #Maximum possible size for the vectors. Trimmed later
    X = np.zeros([length,lookback,nFeatures]) # input to LSTM should be [samples,time_steps,features]
    Y = np.zeros(length)
    counter = 0
    for index in indexes:
        video = videosList[index]
        labels = ysegBasic(labelsList[index] - 1)
        for sample in range(0,video.shape[0],lookback - overlap):
            if sample + lookback <= video.shape[0]:
                #Mold the entry in small batches of LOOKBACK
                currentMiniBatch = np.zeros([lookback,nFeatures])
                for i in range(0,lookback):
                    currentMiniBatch[i,:] = video[sample+i,:]
                X[counter,:,:] = currentMiniBatch
                currentLabelMiniBatch = labels[sample:sample+lookback]
                Y[counter] = currentLabelMiniBatch[-1]
                counter += 1
    
    #Trim X and Y to correct size
    X = X[:counter,:,:]
    Y = Y[:counter];
    return X,Y