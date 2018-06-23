import os
import glob

_VID_DIR = "segs/"
_OUT_DIR = "segs_rgb/"
output_res = (224, 224)

# Load video
for v in glob.glob(_VID_DIR + "*"):
    vname = v.split("/")[-1]
    # Remove the .avi and add pose.mp4
    output_name = vname[:-4]
    if not os.path.exists(_OUT_DIR + output_name):
        os.makedirs(_OUT_DIR + output_name)
    # Remove the .avi and add pose.mp4
    # ffmpeg -i in.avi -vf select='eq(n\,25)+eq(n\,45)' -vsync 0 -s 224x224 frames%d.jpg
    cmd = "ffmpeg -i " + v + " -vf select='eq(n\,25)+eq(n\,35)' -vsync 0 -s 224x224 " + _OUT_DIR + output_name + "/frames%d.jpg"
    print(cmd)
    os.system(cmd)