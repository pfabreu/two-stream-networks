import glob
import os
_DATA_DIR = "/media/pedro/actv3/AHA/videos/original/"
_OUT_DIR = "/media/pedro/actv3/AHA/rgb/"

output_res = (480, 480)
for v in glob.glob(_DATA_DIR + "*"):

    vname = v.split("/")[-1]
    output_name = vname[:-4]
    videoPath = _DATA_DIR + vname
    if not os.path.exists(videoPath):
        print(videoPath + " not found")
    else:
        if not os.path.exists(_OUT_DIR + output_name):
            os.makedirs(_OUT_DIR + output_name)
        print _OUT_DIR + output_name
        # Since no frames are specified this will extract all frames :)
        cmd = "ffmpeg -loglevel panic -i " + videoPath + " -vsync 0 -s " + str(output_res[0]) + "x" + str(output_res[1]) + " " + _OUT_DIR + output_name + "/frames%d.jpg"
        os.system(cmd)
