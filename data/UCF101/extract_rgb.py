import os
import glob

_DATA_DIR = "/media/pedro/actv5/UCF101/videos/"
_OUT_DIR = "/media/pedro/actv5/UCF101/rgb/"
class_folders = glob.glob(_DATA_DIR + "*")
for c in class_folders:
    cname = c.split("/")[-1]
    vids = glob.glob(c + "/*")
    for v in vids:
        # Remove .avi
        vname = v.split("/")[-1]
        vname = vname[:-4]

        output_path = _OUT_DIR + cname + "/" + vname
        print output_path
        if not os.path.exists(output_path):
            print v
            os.makedirs(output_path)
            cmd = "ffmpeg -loglevel 255 -i " + v + " -vf fps=30 -s 224x224 " + output_path + "/frames%d.jpg"
            os.system(cmd)
