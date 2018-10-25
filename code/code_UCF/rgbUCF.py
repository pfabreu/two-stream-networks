from keras.utils import to_categorical, multi_gpu_model
from keras import backend as K
import keras
import utils as utils
from rgbUCF_model import rgb_create_model, compile_model
from rgbUCF_data import load_set


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
              "n_classes": len(classes), "n_channels": 3,
              "shuffle": False, "nb_epochs": 50, "model": "resnet50"}

    # Paths I need for data Load
    annot_path = root_dir_info + "pyannot.pkl"
    trainlist_path = root_dir_info + "trainlistfull.pkl"
#    vallist_path = root_dir_info + "vallist.pkl"
    data_paths_dict = {"RGB": root_dir_data + "UCF_rgb/",
                       "crop": root_dir_data + "UCF_crop/",
                       "fovea": root_dir_data + "UCF_fovea/",
                       "gauss": root_dir_data + "UCF_gauss/"}

    # Load X and Y for Training set
    print("Now loading the Training Set.\n")
    X_Train, y_Train = load_set(annot_path, trainlist_path, data_paths_dict["RGB"], params["dim"], params["n_channels"])
    y_Train = to_categorical(y_Train, params["n_classes"])
    print("Loaded " + str(len(y_Train)) + " data samples.\n")
#    print("Now loading the Validation Set. \n")
#    X_Val, y_Val = load_set(annot_path,vallist_path,data_paths_dict["fovea"], params["dim"], params["n_channels"])
#    y_Val = to_categorical(y_Val,params["n_classes"])
#    print("Loaded " + str(len(y_Val)) + " data samples.\n")

    # Load X and Y for Val Set
    # Create + compile model, load saved weights if they exist
    # saved_weights = None
    saved_weights = "../models/keras-ucf101-rgb-resnet50-split1.hdf5"  # "saved_models/rgb_fovea_resnet50_1806301953.hdf5"
    model, keras_layer_names = rgb_create_model(params["n_classes"], model_name=params['model'], freeze_all=False, conv_fusion=False)
    model = compile_model(model, soft_sigmoid=False)
#    print("Loading MConvNet weights: ")
#    ucf_weights = utils.loadmat("../models/ucf101-img-resnet-50-split1.mat")
#    utils.convert_resnet(model, ucf_weights)
#    model.save(saved_weights)

    if saved_weights is not None:
        model.load_weights(saved_weights)

    # Start the TRAINING
    bestModelPath = "saved_models/rgb_UCF_RGB.hdf5"
    saveBestModel = keras.callbacks.ModelCheckpoint(bestModelPath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')


#    history = model.fit(X_Train, y_Train, batch_size=params['batch_size'], epochs=params["nb_epochs"], verbose=1,validation_data = (X_Val, y_Val),callbacks = [saveBestModel])
    history = model.fit(X_Train, y_Train, batch_size=params['batch_size'], epochs=params["nb_epochs"], verbose=1, shuffle=True, validation_split=0.25, callbacks=[saveBestModel])

    if sendmail:
        utils.sendemail(from_addr='pythonscriptsisr@gmail.com',
                        to_addr_list=['pedro_abreu95@hotmail.com',
                                      'joaogamartins@gmail.com'],
                        cc_addr_list=[],
                        subject='Finished training UCF crop Resnet 50 RGB-stream ',
                        message='Training RGB with following params: ' +
                        str(params),
                        login='pythonscriptsisr@gmail.com',
                        password='1!qwerty')


if __name__ == '__main__':
    main()
