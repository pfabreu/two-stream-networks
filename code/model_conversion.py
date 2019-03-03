    # Create + compile model, load saved weights if they exist
    saved_weights = "../models/ucf_keras/keras-ucf101-rgb-resnet50-sigmoids.hdf5"
    model, keras_layer_names = rgb_create_model(classes=classes['label_id'], loss=loss, model_name=params['model'], freeze_all=params['freeze_all'], conv_fusion=params['conv_fusion'])
    model = compile_model(model, loss=loss)

    initialization = True  # Set to True to use initialization
    kinetics_weights = "a"
    ucf_weights = None

    if saved_weights is not None:
        model.load_weights(saved_weights)
    else:
        if initialization is True:
            if ucf_weights is None:
                print("Loading MConvNet weights: ")
                if params['model'] == "resnet50":
                    ucf_weights = utils.loadmat("../models/ucf_matconvnet/ucf101-img-resnet-50-split1.mat")
                    utils.convert_resnet(model, ucf_weights)
                    model.save("../models/ucf_keras/keras-ucf101-rgb-resnet50-sigmoids.hdf5")
            if kinetics_weights is None:
                if params['model'] == "inceptionv3":
                    print("Loading Keras weights: ")
                    keras_weights = ["../models/kinetics_keras/tsn_rgb_params_names.pkl", "../models/kinetics_keras/tsn_rgb_params.pkl"]
                    utils.convert_inceptionv3(model, keras_weights, keras_layer_names)
                    model.save("../models/kinetics_keras/keras-kinetics-rgb-inceptionv3.hdf5")
