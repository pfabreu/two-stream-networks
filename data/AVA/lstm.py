#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:09:15 2018

@author: jantunes
"""
import numpy as np
import lstm_utils as uf
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
#from keras.backend import clear_session
import keras.callbacks
import pickle
import os
import hdf5storage
import random
#from keras.utils.vis_utils import plot_model
import time
import tensorflow as tf

NFEATURES = 63 
NTRAININGVIDEOS = 39
NTESTINGVIDEOS = 20
NVALIDATIONVIDEOS = 20
LOOKBACK = 30
OVERLAP = 25
NEPOCHS = 300
NHIDDENUNITSV = [32,64,128]#,256,512]
NLABELS = 4
SKPATH = 'Skeletons/'
LABELSPATH = 'Labels/'
NSEEDS = 100
testScores = np.zeros([len(NHIDDENUNITSV),NSEEDS])



print("Loading Data to Memory")
#counterTraining = 0
#counterVideosTrain = 0
#counterVal = 0
#counterVideosVal = 0
videosList = []
labelsList = []
for root, dirs, files in os.walk('Skeletons'):
    for file in sorted(files):
        #Load all files to memory
        skFilename = SKPATH + file;
        labelFilename = LABELSPATH + file[:-8] + 'Labels.mat';
        currentVideo = hdf5storage.loadmat(skFilename);
        currentVideo = np.array(currentVideo['skeletons'][:,:63]);
        videosList.append(currentVideo);
        currentLabel = hdf5storage.loadmat(labelFilename);
        currentLabel = np.array(currentLabel['labels']);
        labelsList.append(currentLabel);
maxLength = 0
for video in videosList:
    if len(video) > maxLength:
        maxLength = len(video)
for ran in range(NSEEDS):
    print("Run #" + str(ran+1) + " of " + str(NSEEDS) + ".")
    indexes = np.array(range(len(videosList)))
    random.Random().shuffle(indexes)
    trainIndexes = np.array(indexes[:NTRAININGVIDEOS]).flatten()
    valIndexes = np.array(indexes[NTRAININGVIDEOS:NTRAININGVIDEOS+NVALIDATIONVIDEOS]).flatten()
    testIndexes = np.array(indexes[-NTESTINGVIDEOS:]).flatten()
    
    #Generate X_Train,Y_Train, X_Val, Y_Val
    X_Train, Y_Train = uf.prepareBatch(trainIndexes,videosList,labelsList,LOOKBACK,OVERLAP,NFEATURES,maxLength)
    X_Val, Y_Val = uf.prepareBatch(valIndexes,videosList,labelsList,LOOKBACK,OVERLAP,NFEATURES,maxLength)
    Y_Train1Hot = to_categorical(Y_Train,2)
    Y_Val1Hot = to_categorical(Y_Val,2)
    for huIndex in range(len(NHIDDENUNITSV)):
        NHIDDENUNITS = NHIDDENUNITSV[huIndex];
        #NHIDDENUNITS2 = NHIDDENUNITS2V[index]
        print( "Training the model with " + str(NHIDDENUNITS) + "HU.")
        with tf.device('/gpu:0'):
            start = time.time()
            model = Sequential()
            model.add(LSTM(NHIDDENUNITS,input_shape = (LOOKBACK,NFEATURES),activation = "tanh",use_bias = True))#, return_sequences = True))
            #model.add(LSTM(NHIDDENUNITS2,activation = "tanh",use_bias = True))
            model.add(Dense(2,activation = "softmax",use_bias = True))
            model.compile(optimizer = "adam", loss = "categorical_crossentropy",metrics=['accuracy'])

            bestModelPath = "ResultsSeg/Basic/" + str(NHIDDENUNITS) + "HU/bestModel.hdf5";
            maxValScore = 0;
            saveBestModel = keras.callbacks.ModelCheckpoint(bestModelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
            hist = model.fit(X_Train,Y_Train1Hot,batch_size = X_Train.shape[0],epochs = NEPOCHS, validation_data = (X_Val, Y_Val1Hot), verbose = 0,callbacks = [saveBestModel])
            
            #Evaluate the best model on the test Set!
            #Load the best model
            model.load_weights(bestModelPath)
            # Generate X_Test, Y_Test
            X_Test, Y_Test = uf.prepareBatch(testIndexes,videosList,labelsList,LOOKBACK,OVERLAP,NFEATURES,maxLength)
            Y_Test1Hot = to_categorical(Y_Test,2)
            loss,acc = model.evaluate(X_Test,Y_Test1Hot,verbose = 0);
            #print('Best Val Score = ' + str(maxValScore))
            print('Test Score = ' + str(acc))
            print('\n')
            end = time.time()
            print("The model with" + str(NHIDDENUNITS) + "HU took " + str(end-start) + " seconds to train and evaluate")
            testScores[huIndex,ran] = acc;
            keras.backend.clear_session() #KERAS HAS A MEMORY LEAK WHEN BUILDING MORE THAN ONE MODEL!!!
#            
            
print(str(np.mean(testScores)))
#        testScoreString = "ResultsClass/Deriv/" + str(NHIDDENUNITS) + "HU/testScore.txt"
#        valScoreString = "ResultsClass/Deriv/" + str(NHIDDENUNITS) + "HU/valScore.txt"
#        resultString = "ResultsClass/Deriv/" + str(NHIDDENUNITS) + "HU/results.txt"
#        f = open(testScoreString,'wb')
#        pickle.dump(str(testScore),f)
#        f.close();

resultsMean = np.mean(testScores,1)
finalResultsString = "ResultsSeg/Basic/AllModelsResults.txt"
f = open(finalResultsString,'wb')
pickle.dump(testScores,f)
f.close();
finalResultsString = "ResultsSeg/Basic/AllModelsMean.txt"
f = open(finalResultsString,'wb')
pickle.dump(resultsMean,f)
f.close();

#        with open(resultString, "w") as text_file:
#            print("The model with {}/{} HU got:\nFinal Test Score: {}\nBest Validation Score: {}\nIt took {} seconds to train.".format(NHIDDENUNITS,NHIDDENUNITS,testScore,maxValScore,(end-start)), file=text_file)

    
#    Ystring = "ResultsClassificationLookback/" + str(NHIDDENUNITS) + "HU/Y.txt"
#    YPredstring = "ResultsClassificationLookback/" + str(NHIDDENUNITS)  + "HU/Yhat.txt"
#    f = open(Ystring,'w')
#    pickle.dump(Y_Test,f)
#    f.close();
#    f = open(YPredstring,'w')
#    pickle.dump(Yhat_Test,f)
#    f.close();