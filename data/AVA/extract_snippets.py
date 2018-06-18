import cv2
import os
import glob
import datetime

_VID_DIR = "/media/pedro/actv4/AVA/Videos_test/"
set_type = "test"
_OUTPUT_DIR = "/media/pedro/actv4/AVA/snippets_" + set_type + "/"

for v in glob.glob(_VID_DIR + "*"):
    vname = v.split("/")[-1]
    path_splits = v.split("/")[:-1]
    if ".webm" in vname:
        output_name = vname[:-5]
    if ".mkv" in vname or ".mp4" in vname:
        output_name = vname[:-4]
    vidname = output_name + ".avi"
    if not os.path.exists(_OUTPUT_DIR + vidname):

        vidcap = cv2.VideoCapture(v)
        vid_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        lower_bound = 15 * 60
        upper_bound = 30 * 60

        success, test_image = vidcap.read()
        height, width, layers = test_image.shape
        success = True

        start_time = str(datetime.timedelta(seconds=lower_bound))
        end_time = str(datetime.timedelta(seconds=upper_bound))
        out_vid = _OUTPUT_DIR + vidname
        print out_vid
        os.system("ffmpeg -loglevel panic -i " + v + " -ss " +
                  start_time + " -to " + end_time + " " + out_vid)
