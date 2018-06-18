import os
import glob

_VID_DIR = "/media/pedro/actv3/AHA/pose/"
_OUT_DIR = "/media/pedro/actv3/AHA/pose_resized/"
output_res = (448, 448)
output_fps = 30
util = "spatial"  # Must be spatial or temporal

# Load video
for v in glob.glob(_VID_DIR + "*"):
    vname = v.split("/")[-1]
    path_splits = v.split("/")[:-1]
    # Remove the .avi and add pose.mp4
    output_name = vname[:-4]
    if util == "spatial":
        os.system("ffmpeg -i " + v + " -s " + str(output_res[0]) + "x" + str(output_res[1]) + " -c:a copy " + _OUT_DIR + vname)
    elif util == "temporal":
        os.system("ffmpeg -y -i " + v + " -r " + str(output_fps) + " " + _OUT_DIR + vname)
