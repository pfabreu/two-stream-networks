import os
import glob
import cv2

_IN_DIR = "/media/pedro/actv4/AVA/pose/"
_VID_DIR = "/media/pedro/actv4/AVA/Videos/"
_OUT_DIR = "/media/pedro/actv4/AVA/pose_fps/"
output_fps = 30

# Load video
for v in glob.glob(_VID_DIR + "*"):
    print v
    vidcap = cv2.VideoCapture(v)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps < 30:
        rate = 30 / fps
        vname = v.split("/")[-1]
        path_splits = v.split("/")[:-1]
        if ".webm" in vname:
            output_name = vname[:-5]
        if ".mkv" in vname or ".mp4" in vname:
            output_name = vname[:-4]
        vp = _IN_DIR + output_name + ".avi"
        if os.path.exists(vp):
            print vname
            if not os.path.exists(_OUT_DIR + output_name + ".avi"):

                # os.system("ffmpeg -y -i " + v + " -r " +
                #          str(output_fps) + " " + _OUT_DIR + vname)
                cmd = "ffmpeg -y -i " + vp + " -filter:v \"setpts=" + str(rate) + "*PTS\" " + _OUT_DIR + output_name + ".avi"
                print cmd
                os.system(cmd)
