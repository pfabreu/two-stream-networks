from keras.utils import to_categorical, multi_gpu_model
from keras import backend as K
import keras
import utils as utils
from flowUCF_model import flow_create_model, compile_model
from flowUCF_data import load_set, make_chunks
import pickle
import math
import gc


def main():
    # Erase previous models from GPU memory
    root_dir_info = "../../data/UCF101/"
    root_dir_data = "../../../"

    K.clear_session()

    sendmail = True

    # Load list of action classes and separate them
    with open(root_dir_info + "classInd.txt") as f:
        classes = f.readlines()

    # Parameters for training
    params = {"dim": (224, 224), "batch_size": 32,
              "n_classes": len(classes), "n_channels_rgb": 3,
              "n_channels_of": 20, "of_length": 10,
              "shuffle": False, "nb_epochs": 50, "model": "resnet50",
              "train_chunk_size": 300}

    # Paths I need for data Load
    annot_path = root_dir_info + "pyannot.pkl"
    index_path = root_dir_info + "indexesDict.pkl"
    trainlist_path = root_dir_info + "trainlistfull.pkl"
    with open(trainlist_path, 'rb') as f:
        setlist = pickle.load(f)
    trainsamples = len(setlist) * 0.75
    trainlist = setlist[:math.floor(trainsamples)]
    vallist = setlist[math.floor(trainsamples):]
    chunks = make_chunks(setlist, len(setlist), params["train_chunk_size"])
    valchunks = make_chunks(vallist, len(vallist), params["train_chunk_size"])
    data_paths_dict = {"RGB": root_dir_data + "UCF_rgb/",
                       "crop": root_dir_data + "UCF_crop/",
                       "fovea": root_dir_data + "UCF_fovea/",
                       "gauss": root_dir_data + "UCF_gauss/",
                       "flow": root_dir_info + "tvl1_flow"}

    saved_weights = "../models/keras-ucf101-TVL1flow-resnet50-split1.hdf5"  # "saved_models/rgb_fovea_resnet50_1806301953.hdf5"
    model, keras_layer_names = flow_create_model(params["n_classes"], model_name=params['model'], freeze_all=False, conv_fusion=False)
    model = compile_model(model, soft_sigmoid=False)

#    print("Loading MConvNet weights: ")
#    ucf_weights = utils.loadmat("../models/ucf101-TVL1flow-resnet-50-split1.mat")
#    utils.convert_resnet(model, ucf_weights)
#    model.save(saved_weights)

    if saved_weights is not None:
        model.load_weights(saved_weights)

    # Start the TRAINING
    bestModelPath = "saved_models/rgb_UCF_flow.hdf5"
    saveBestModel = keras.callbacks.ModelCheckpoint(bestModelPath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    maxValAcc = 0
    for epoch in range(params["nb_epochs"]):
        print("Now on epoch " + str(epoch) + ".\n")
        epochAccuracies = []
        for chunk in chunks:
            X_Train = y_Train = X_Val = y_Val = None
            X_Train, y_Train = load_set(annot_path, chunk, index_path, data_paths_dict["flow"], params["dim"], params["n_channels_of"], params["of_length"])
            y_Train = to_categorical(y_Train, params["n_classes"])
            history = model.fit(X_Train, y_Train, batch_size=params['batch_size'], epochs=1, verbose=1, shuffle=True)
            gc.collect()
        for chunk in valchunks:
            X_Train = y_Train = X_Val = y_Val = None
            X_Val, y_Val = load_set(annot_path, chunk, index_path, data_paths_dict["flow"], params["dim"], params["n_channels_of"], params["of_length"])
            y_Val = to_categorical(y_Val, params["n_classes"])
            loss, acc = model.evaluate(X_Val, y_Val, batch_size=params["batch_size"], verbose=0)
            epochAccuracies.append(acc)
        currentValAcc = sum(epochAccuracies) / len(epochAccuracies)
        if currentValAcc > maxValAcc:
            maxValAcc = currentValAcc
            model.save(bestModelPath)
            print("New best model. Best val_acc is now " + str(maxValAcc) + ".\n")
    print("TODO: Fix it so chunks doesn't discard any data and val accuracy calculator takes chunk sizes into account.")

if __name__ == '__main__':
    main()
