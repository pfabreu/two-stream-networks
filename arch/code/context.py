from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from context_model import context_create_model, compile_model
from context_data import load_split, get_AVA_set, get_AVA_labels

import pickle
import utils


def main():
    # Load list of action classes and separate them (from utils_stream)
    classes = utils.get_AVA_classes('ava_action_list_custom.csv')

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': 270, 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 1,
              'shuffle': False, 'nb_epochs': 200, 'model': "mlp", 'email': True}

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['train'] = get_AVA_set(classes=classes, filename="AVA_train_Custom_Corrected.csv")  # IDs for training
    partition['validation'] = get_AVA_set(classes=classes, filename="AVA_validation_Custom_Corrected.csv")  # IDs for validation

    # pprint.pprint(partition['validation'])
    # pprint.pprint(partition['validation'])

    # Labels
    labels_train = get_AVA_labels(classes, partition, "train", filename="AVA_train_Custom_Corrected.csv")
    labels_val = get_AVA_labels(classes, partition, "validation", filename="AVA_validation_Custom_Corrected.csv")
    # pprint.pprint(labels_val)

    # Create + compile model, load saved weights if they exist
    NHU1V = [32, 64, 128, 256, 512]
    NHU2V = [16, 32, 64, 128, 256]
    for NHU1, NHU2 in zip(NHU1V, NHU2V):
        bestModelPath = "bestModelContext_" + str(NHU1) + ".hdf5"
        histPath = "contextHistory_" + str(NHU1)
        checkpointer = ModelCheckpoint(filepath=bestModelPath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, period=1)
        model = context_create_model(NHU1, NHU2, in_shape=(params['dim'],))
        model = compile_model(model)

        x_val = y_val_pose = y_val_object = y_val_human = x_train = y_train_pose = y_train_object = y_train_human = None
        x_train, y_train_pose, y_train_object, y_train_human = load_split(partition['train'], labels_train, params['dim'], params['n_channels'], "train")

        y_t = []
        y_t.append(to_categorical(y_train_pose, num_classes=utils.POSE_CLASSES))
        y_t.append(utils.to_binary_vector(y_train_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
        y_t.append(utils.to_binary_vector(y_train_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))

        x_val, y_val_pose, y_val_object, y_val_human = load_split(partition['validation'], labels_val, params['dim'], params['n_channels'], "val")
        y_v = []
        y_v.append(to_categorical(y_val_pose, num_classes=utils.POSE_CLASSES))
        y_v.append(utils.to_binary_vector(y_val_object, size=utils.OBJ_HUMAN_CLASSES, labeltype='object-human'))
        y_v.append(utils.to_binary_vector(y_val_human, size=utils.HUMAN_HUMAN_CLASSES, labeltype='human-human'))

        hist = model.fit(x_train, y_t, batch_size=params['batch_size'], validation_data=(x_val, y_v), epochs=params['nb_epochs'], verbose=1, callbacks=[checkpointer])
        # model.save(bestModelPath)
        with open(histPath, 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)


if __name__ == '__main__':
    main()
