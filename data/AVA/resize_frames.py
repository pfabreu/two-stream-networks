import os
from PIL import Image
import glob

type_resize = "rgb"
set_type = "train"
_DATA_DIR_RGB = "ava_" + set_type + "/rgb/"
#_DATA_DIR_FLOW_X = "/media/pedro/actv4/AVA-split/" + set_type + "/x/"
#_DATA_DIR_FLOW_Y = "/media/pedro/actv4/AVA-split/" + set_type + "/y/"
_OUT_DIR_RGB = "ava_" + set_type + "_resized/rgb/"
#_OUT_DIR_FLOW_X = "/media/pedro/actv4/AVA-split/" + set_type + "_resize/x/"
#_OUT_DIR_FLOW_Y = "/media/pedro/actv4/AVA-split/" + set_type + "_resize/y/"
output_res = (224, 224)
if type_resize == "rgb":
    fcount = 0
    for dirs in os.listdir(_DATA_DIR_RGB):
        print("rgb: " + str(fcount))
        full_dir_path = _DATA_DIR_RGB + dirs + "/"
        for filename in glob.glob(full_dir_path + "*.jpg"):
            # print filename
            frame = filename.rsplit("/", 1)[1]
            savedir = _OUT_DIR_RGB + dirs + "/" + frame
            if not os.path.exists(savedir):
                # print frame
                # Read image
                img = Image.open(filename)
                # Resize
                # Methods are: Image.NEAREST, Image.BILINEAR, Image.BICUBIC,
                # Image.LANCZOS
                img = img.resize(output_res, Image.NEAREST)
                # If output dir doesn't exist, create it
                if not os.path.exists(_OUT_DIR_RGB + dirs + "/"):
                    os.makedirs(_OUT_DIR_RGB + dirs + "/")
                # Save img
                img.save(savedir)
        fcount += 1

#
#if type_resize == "flow":
#    fcount = 0
#    for dirs in os.listdir(_DATA_DIR_FLOW_X):
#        print "flow_x: " + str(fcount)
#        full_dir_path = _DATA_DIR_FLOW_X + dirs + "/"
#        for filename in glob.glob(full_dir_path + "*.jpg"):
#            # print filename
#            frame = filename.rsplit("/", 1)[1]
#            savedir = _OUT_DIR_FLOW_X + dirs + "/" + frame
#            if not os.path.exists(savedir):
#                print "resizing"
#                # print frame
#                # Read image
#                img = Image.open(filename)
#                # Resize
#                # Methods are: Image.NEAREST, Image.BILINEAR, Image.BICUBIC,
#                # Image.LANCZOS
#                img = img.resize(output_res, Image.NEAREST)
#                # If output dir doesn't exist, create it
#                if not os.path.exists(_OUT_DIR_FLOW_X + dirs + "/"):
#                    os.makedirs(_OUT_DIR_FLOW_X + dirs + "/")
#
#                # Save img
#                img.save(savedir)
#        fcount += 1
#    fcount = 0
#    for dirs in os.listdir(_DATA_DIR_FLOW_Y):
#        print "flow_y: " + str(fcount)
#        full_dir_path = _DATA_DIR_FLOW_Y + dirs + "/"
#        for filename in glob.glob(full_dir_path + "*.jpg"):
#            # print filename
#            frame = filename.rsplit("/", 1)[1]
#            savedir = _OUT_DIR_FLOW_Y + dirs + "/" + frame
#            if not os.path.exists(savedir):
#                print "resizing"
#                # print frame
#                # Read image
#                img = Image.open(filename)
#                # Resize
#                # Methods are: Image.NEAREST, Image.BILINEAR, Image.BICUBIC,
#                # Image.LANCZOS
#                img = img.resize(output_res, Image.NEAREST)
#                # If output dir doesn't exist, create it
#                if not os.path.exists(_OUT_DIR_FLOW_Y + dirs + "/"):
#                    os.makedirs(_OUT_DIR_FLOW_Y + dirs + "/")
#
#                # Save img
#                img.save(savedir)
#        fcount += 1
