import os
import glob

_VID_DIR = "/media/pedro/actv4/AVA/trimmed_videos/"
_OUT_DIR = "/media/pedro/actv4/AVA/trimmed_videos_fps/"
output_fps = 30

# Load video
for v in glob.glob(_VID_DIR + "*"):
    vname = v.split("/")[-1]
    path_splits = v.split("/")[:-1]
    # Remove the .avi and add pose.mp4
    output_name = vname[:-4]

    os.system("ffmpeg -y -i " + v + " -r " +
              str(output_fps) + " " + _OUT_DIR + vname)
