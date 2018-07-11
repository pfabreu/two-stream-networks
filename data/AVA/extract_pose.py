import cv2
import os
import math
import glob

set_type = 'test'
_VID_DIR = '/media/pedro/actv-ssd/segments_' + set_type + '/'
_OUT_DIR = "/media/pedro/actv-ssd/pose_" + set_type + "/"
OPENPOSE_DIR = "/home/pedro/openpose/"
MODELS_DIR = "../../arch/models/openpose_models/"

# Load video
vid_count = 0
vids = glob.glob(_VID_DIR + "*")
continue_idx = 0  # set to 0 if you want ot process all videos

for v in vids:
    if vid_count >= continue_idx:
        print(v)
        vname = v.split("/")[-1]
        path_splits = v.split("/")[:-1]
        # Remove the .avi and add pose.mp4
        output_name = vname[:-4]
        print "Video: " + output_name + "  " + str(vid_count) + "/" + str(len(vids))
        if not os.path.exists(_OUT_DIR + output_name + ".avi"):
            vidcap = cv2.VideoCapture(v)

            success, test_image = vidcap.read()
            height, width, layers = test_image.shape
            width_rounded = int(math.ceil(width / 16.0) * 16)
            height_rounded = int(math.ceil(height / 16.0) * 16)

            # Make folder with options
            face = True
            hand = True
            input_video = _VID_DIR + vname
            output_video = _OUT_DIR + output_name + "_orig.avi"
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
                str(lv) + " --disable_blending 1 --display 0 --video " + input_video + \
                " --write_video " + output_video + " --write_json " + output_joint_path + " --model_folder " + MODELS_DIR
            if face:
                cmd += " --face"
            if hand:
                cmd += " --hand"
            os.system(cmd)

            # Rescale video
            ffmpeg_cmd = "ffmpeg -loglevel " + str(0) + " -i " + output_video + " -vf scale=224:224 " + _OUT_DIR + output_name + ".avi"
            os.system(ffmpeg_cmd)

    vid_count += 1
