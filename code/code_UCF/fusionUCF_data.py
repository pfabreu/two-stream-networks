"""
Class for managing our data.
"""
import csv
import numpy as np
import cv2
import os.path
import sys
#import utils
import pickle


def make_chunks(original_list, size, chunk_size):
    seq = original_list[:size]
    splits = [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]
    return splits


def load_set(annot_path, setlist, index_path, flowdatadir, rgbdatadir, dim, n_channels_rgb, n_channels_of, of_length):
    X_rgb = np.zeros([2000, dim[0], dim[1], n_channels_rgb])  # overshooting size. will cut to right size at the end of this function.
    X_flow = np.empty([2000, dim[0], dim[1], n_channels_of])  # overshooting size. will cut to right size at the end of this function.
    y = np.empty(2000)  # overshooting size. will cut to right size at the end of this function.

    'Load FLOW'
    # Load annotations and set list
    with open(annot_path, 'rb') as f:
        annots = pickle.load(f)
    with open(index_path, 'rb') as f:
        frame_indexes = pickle.load(f)
    keys = []
    for st in setlist:
        keys.append(st.split(' ')[0][:-4])
    flowdataIndex = 0
    for key in keys:
        if key in annots:
            dirinfo = key.split('/')
            for BB in range(len(annots[key]['annotations'])):
                index_key = dirinfo[1] + "_BB_" + str(BB)
                current_indexes = frame_indexes[index_key]
                udir = flowdatadir + "/u/" + dirinfo[1] + "/"
                vdir = flowdatadir + "/v/" + dirinfo[1] + "/"
                label = annots[key]['annotations'][BB]['label']
                for img_number in current_indexes:
                    # Load the 20 images corresponding to the current rgb frame
                    v = 0
                    of_volume = np.zeros([dim[0], dim[1], n_channels_of])
                    for fn in range(-of_length // 2, of_length // 2):
                        current_frame = img_number + fn
                        uimg_path = udir + "frame" + str(current_frame).zfill(6) + ".jpg"
                        vimg_path = vdir + "frame" + str(current_frame).zfill(6) + ".jpg"
                        if not os.path.exists(uimg_path):
                            u_img = np.zeros([dim[0], dim[1]])
                        else:
                            u_img = cv2.resize(cv2.imread(uimg_path, cv2.IMREAD_GRAYSCALE), (dim[0], dim[1]))
                        if not os.path.exists(vimg_path):
                            v_img = np.zeros([dim[0], dim[1]])
                        else:
                            v_img = cv2.resize(cv2.imread(vimg_path, cv2.IMREAD_GRAYSCALE), (dim[0], dim[1]))
                        of_volume[:, :, v] = u_img
                        v += 1
                        of_volume[:, :, v] = v_img
                        v += 1
                    X_flow[flowdataIndex, ] = of_volume
                    y[flowdataIndex] = label
                    flowdataIndex += 1

    'Load RGB'
    rgbdataIndex = 0
    for key in keys:
        if key in annots:
            dirinfo = key.split('/')
            for BB in range(len(annots[key]['annotations'])):
                if rgbdatadir == "../../../UCF_rgb/":
                    frame_path = dirinfo[1] + "/"
                else:
                    frame_path = dirinfo[1] + "_BB_" + str(BB) + "/"
                label = annots[key]['annotations'][BB]['label']
                for img_number in range(1, 6):
                    # Load the 5 images!
                    img_path = rgbdatadir + frame_path + "frame" + str(img_number) + ".jpg"
                    if not os.path.exists(img_path):
                        print(img_path)
                        print("[Error] File does not exist!")

                    else:
                        img = cv2.imread(img_path)
                        X_rgb[rgbdataIndex, ] = img
                        rgbdataIndex = rgbdataIndex + 1

        else:
            #print(key + " is not in our annotations. Skipping!\n")
            pass

    # Clip X and y to correct lengths
    if rgbdataIndex < flowdataIndex:
        dataIndex = rgbdataIndex
    else:
        dataIndex = flowdataIndex
    X_rgb = X_rgb[:dataIndex, :, :, :]
    X_flow = X_flow[:dataIndex, :, :, :]
    y = y[:dataIndex]

    return X_rgb, X_flow, y
