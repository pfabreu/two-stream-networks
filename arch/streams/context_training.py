import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
import utils
from context_training_model import context_create_model, compile_model
from context_training_data import load_split, get_AVA_classes, get_AVA_set, get_AVA_labels
from keras.callbacks import ModelCheckpoint
import pprint
import sys


def to_binary_vector(list_classes, size, labeltype):
    """
    Keras should have this function...
    """
    POSE_CLASSES = 14
    OBJ_HUMAN_CLASSES = 49
    # HUMAN_HUMAN_CLASSES = 17
    labelsarray = np.empty([len(list_classes), size])
    offset = 0
    if labeltype == 'object-human':
        offset = POSE_CLASSES
    elif labeltype == 'human-human':
        offset = POSE_CLASSES + OBJ_HUMAN_CLASSES
    elif labeltype == 'pose':
        offset = 0
    index = 0
    for l in list_classes:
        bv = np.zeros(size)
        lv = l
        if len(lv) >= 1:
            lv = [x - offset for x in lv]
        for c in lv:
            v = to_categorical(c, size)
            bv = bv + v

        labelsarray[index, :] = bv
        index += 1
    return labelsarray

    
def main():
    sendmail = True
    # Load list of action classes and separate them (from utils_stream)
    classes = get_AVA_classes('AVA2.1/ava_action_list_v2.1.csv')
    checkpointer = ModelCheckpoint(filepath="bestModelcontext128.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, period=1)

    # Parameters for training (batch size 32 is supposed to be the best?)
    params = {'dim': 720, 'batch_size': 64,
              'n_classes': len(classes['label_id']), 'n_channels': 1,
              'shuffle': False, 'nb_epochs': 50, 'model': "mlp"}

    # Get ID's and labels from the actual dataset
    partition = {}
    partition['train'] = get_AVA_set(
        classes=classes, filename="AVA2.1/ava_mini_split_train_big.csv")  # IDs for training
    partition['train'] = list(set(partition['train']))  # Make sure all IDs are unique
    partition['validation'] = get_AVA_set(
        classes=classes, filename="AVA2.1/ava_mini_split_val_big.csv")  # IDs for validation
    print(len(partition['validation']))
    partition['validation'] = list(set(partition['validation']))  # Make sure all IDs are unique
    print(len(partition['validation']))
    # pprint.pprint(partition['validation'])
    # pprint.pprint(partition['validation'])
    # Labels
    labels_train = get_AVA_labels(classes, partition, "train", filename="AVA2.1/ava_mini_split_train_big.csv")
    labels_val = get_AVA_labels(classes, partition, "validation", filename="AVA2.1/ava_mini_split_val_big.csv")
    # pprint.pprint(labels_val)

    # Create + compile model, load saved weights if they exist
    saved_weights = None
    model_name = "mlp"
    model = context_create_model(classes=classes['label_id'], model_name=model_name, in_shape=(720,))
    model = compile_model(model)

    x_val = y_val_pose = y_val_object = y_val_human = x_train = y_train_pose = y_train_object = y_train_human = None

    x_train, y_train_pose, y_train_object, y_train_human = load_split(partition['train'], labels_train, params['dim'], params['n_channels'], "train")

    y_t = []
    POSE_CLASSES = 14
    OBJ_HUMAN_CLASSES = 49
    HUMAN_HUMAN_CLASSES = 17
    y_t.append(to_categorical(y_train_pose, num_classes=POSE_CLASSES))
    y_t.append(to_binary_vector(y_train_object, size=OBJ_HUMAN_CLASSES, labeltype='object-human'))
    y_t.append(to_binary_vector(y_train_human, size=HUMAN_HUMAN_CLASSES, labeltype='human-human'))

    x_val, y_val_pose, y_val_object, y_val_human = load_split(partition['validation'], labels_val, params['dim'], params['n_channels'], "val")
    y_v = []
    y_v.append(to_categorical(y_val_pose, num_classes=POSE_CLASSES))
    y_v.append(to_binary_vector(y_val_object, size=OBJ_HUMAN_CLASSES, labeltype='object-human'))
    y_v.append(to_binary_vector(y_val_human, size=HUMAN_HUMAN_CLASSES, labeltype='human-human'))

    hist = model.fit(x_train, y_t, batch_size=params['batch_size'], validation_data=(x_val, y_v), epochs=params['nb_epochs'], verbose=0, callbacks=[checkpointer])
    print("The end!")
    #model.save(bestModelPath)
    # Model with 256HU : Val Loss: 1.18289


if __name__ == '__main__':
    main()
