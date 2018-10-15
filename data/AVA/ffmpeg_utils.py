import os
import glob

_VID_DIR = "/media/pedro/actv4/AVA/trimmed_videos/"
_OUT_DIR = "/media/pedro/actv4/AVA/trimmed_videos_fps/"
ffmpeg_func = "change_size"  # change_fps/change_size
output_fps = 30
output_res = (224, 224)

# Load video
for v in glob.glob(_VID_DIR + "*"):
    vname = v.split("/")[-1]
    path_splits = v.split("/")[:-1]
    # Remove the .avi and add pose.mp4
    output_name = vname[:-4]
    if ffmpeg_func == "change_fps":
        os.system("ffmpeg -y -i " + v + " -r " +
                  str(output_fps) + " " + _OUT_DIR + vname)
    elif ffmpeg_func == "change_size":
        os.system("ffmpeg -i " + v +
                  " -s " + str(output_res[0]) + "x" + str(output_res[1]) + " -c:a copy " + _OUT_DIR + vname)
