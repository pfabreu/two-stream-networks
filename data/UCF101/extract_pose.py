import cv2
import os
import math
import glob

_VID_DIR = '/media/pedro/actv5/UCF101/videos/'
_OUT_DIR = "/media/pedro/actv5/UCF101/pose/"
OPENPOSE_DIR = "/home/pedro/openpose/"
MODELS_DIR = "../../arch/models/openpose_models/"

# Load video
vid_count = 0
class_folders = glob.glob(_VID_DIR + "*")
continue_idx = 0  # set to 0 if you want ot process all videos

for c in class_folders:
    vids = glob.glob(c + "/*")
    clas = c.rsplit("/", 1)
    print clas
    if not os.path.exists(_OUT_DIR + clas[1]):
        os.makedirs(_OUT_DIR + clas[1])
    for v in vids:
        if vid_count >= continue_idx:
            vname = v.split("/")[-1]
            path_splits = v.split("/")[:-1]
            # Remove the .avi and add pose.mp4
            output_name = vname[:-4]
            if not os.path.exists(_OUT_DIR + clas[1] + "/" + output_name + ".avi"):
                print(v)
                vidcap = cv2.VideoCapture(v)

                success, test_image = vidcap.read()
                height, width, layers = test_image.shape
                width_rounded = int(math.ceil(width / 16.0) * 16)
                height_rounded = int(math.ceil(height / 16.0) * 16)

                print "Video: " + output_name + "  " + str(vid_count) + "/" + str(len(vids))
                # Make folder with options
                face = True
                hand = True
                input_video = _VID_DIR + vname
                output_video = _OUT_DIR + output_name + ".avi"
                output_joint_path = _OUT_DIR + "joints/" + output_name + "/"
                # Make output_joint_path if it doesnt exist
                if not os.path.exists(output_joint_path):
                    os.makedirs(output_joint_path)
                model = 'MPI'
                # input_res should be video size
                max_res = (1200, 700)
                input_res = (width_rounded, height_rounded)
                output_res = (224, 224)
                lv = 255

                cmd = OPENPOSE_DIR + "./build/examples/openpose/openpose.bin --logging_level " + \
                    str(lv) + " --disable_blending 1 --display 0 --video " + v + \
                    " --write_video " + _OUT_DIR + clas[1] + "/" + output_name + ".avi" + " --write_json " + " --model_folder " + MODELS_DIR
                if face:
                    cmd += " --face"
                if hand:
                    cmd += " --hand"
                os.system(cmd)
        vid_count += 1
