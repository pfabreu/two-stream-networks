from keras.utils import to_categorical, multi_gpu_model
from keras import backend as K
import keras
import utils as utils
from fusionUCF_model import TwoStreamModel
from fusionUCF_data import load_set, make_chunks
import pickle
import math
import gc
def main():
    # Erase previous models from GPU memory
    root_dir_info = "../../data/UCF101/"
    root_dir_data = "../../../"

    K.clear_session()

    sendmail = True
    attention_type = "gauss"

    # Load list of action classes and separate them
    with open(root_dir_info + "classInd.txt") as f:
        classes = f.readlines()

    # Parameters for training
    params = {"dim": (224, 224), "batch_size": 32,
              "n_classes": len(classes), "n_channels_rgb": 3,
              "n_channels_of": 20, "of_length": 10,
              "shuffle": False, "nb_epochs": 50, "model": "resnet50",
              "train_chunk_size": 200}

    #Paths I need for data Load
    annot_path = root_dir_info + "pyannot.pkl"
    index_path = root_dir_info + "indexesDict.pkl"
    trainlist_path = root_dir_info + "trainlistfull.pkl"
    with open(trainlist_path, 'rb') as f:
        setlist = pickle.load(f)
    trainsamples = len(setlist)*0.75
    trainlist = setlist[:math.floor(trainsamples)]
    vallist = setlist[math.floor(trainsamples):]
    chunks = make_chunks(setlist, len(setlist), params["train_chunk_size"])
    valchunks = make_chunks(vallist,len(vallist), params["train_chunk_size"])
    data_paths_dict = {"RGB" : root_dir_data + "UCF_rgb/",
                       "crop" : root_dir_data + "UCF_crop/",
                       "fovea" : root_dir_data + "UCF_fovea/",
                       "gauss" : root_dir_data + "UCF_gauss/",
                       "flow" : root_dir_info + "tvl1_flow"}

    flow_saved_weights = "saved_models/flow_UCF.hdf5" #"saved_models/rgb_fovea_resnet50_1806301953.hdf5"
    rgb_saved_weights = "saved_models/rgb_UCF_" + attention_type + ".hdf5"
    
    nsmodel = TwoStreamModel(params["n_classes"], rgb_saved_weights, flow_saved_weights)
    nsmodel.compile_model(soft_sigmoid = False)
    model = nsmodel.model


    #Start the TRAINING
    bestModelPath = "saved_models/UCF_fusion_"+ attention_type + ".hdf5"
#    saveBestModel = keras.callbacks.ModelCheckpoint(bestModelPath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    maxValAcc = 0
    for epoch in range(params["nb_epochs"]):
        print("Now on epoch " + str(epoch) + ".\n")
        epochAccuracies = []
        epochValLengths = []
        for chunk in chunks:
            X_Train_RGB = X_Train_OF = y_Train = X_Val_RGB = X_Val_OF = y_Val = None
            X_Train_RGB, X_Train_OF, y_Train = load_set(annot_path,chunk,index_path,data_paths_dict["flow"],data_paths_dict[attention_type],params["dim"],params["n_channels_rgb"],params["n_channels_of"],params["of_length"])
            y_Train = to_categorical(y_Train,params["n_classes"])
            history = model.fit([X_Train_RGB,X_Train_OF], y_Train, batch_size=params['batch_size'], epochs=1, verbose=1,shuffle = True)
            gc.collect()
        for chunk in valchunks:
            X_Train_RGB = X_Train_OF = y_Train = X_Val_RGB = X_Val_OF = y_Val = None
            X_Val_RGB, X_Val_OF, y_Val = load_set(annot_path,chunk,index_path,data_paths_dict["flow"],data_paths_dict[attention_type],params["dim"],params["n_channels_rgb"],params["n_channels_of"],params["of_length"])
            y_Val = to_categorical(y_Val,params["n_classes"])
            loss,acc = model.evaluate([X_Val_RGB,X_Val_OF], y_Val, batch_size = params["batch_size"], verbose = 0)
            epochAccuracies.append(acc)
            epochValLengths.append(len(y_Val))
        currentValAcc = 0
        for i in range(len(epochAccuracies)):
            currentValAcc += epochAccuracies[i] * epochValLengths[i]
        currentValAcc = currentValAcc/sum(epochValLengths)
        if currentValAcc > maxValAcc:
            maxValAcc = currentValAcc
            model.save(bestModelPath)
            print("New best model. Best val_acc is now " + str(maxValAcc) + ".\n")
    print("Max Val Acc achieved was: " + str(maxValAcc))

if __name__ == '__main__':
    main()