import csv
import os
import sys
import glob

set_type = "val"
use_glob = False
_DATA_DIR = "/media/pedro/actv-ssd/pose_noface_" + set_type + "/"
_OUT_DIR = "/media/pedro/actv-ssd/pose_rgb_noface/" + set_type + "/"

snippets_video = []
snippets_time = []


with open('files/AVA_' + set_type.title() + '_Custom_Corrected.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        snippets_video.append(row[0])
        snippets_time.append(row[1])

# Find unique segments

currentVideo = ''
currentTime = ''
if use_glob is False:
    for i in range(len(snippets_video)):
        if i % 100 == 0:
            print("Video #" + str(i) + " of " + str(len(snippets_video)) + ".")
        if currentVideo == snippets_video[i] and currentTime == snippets_time[i]:
            pass
        else:
            currentVideo = snippets_video[i]
            currentTime = snippets_time[i]
            # print(currentTime)

            if currentTime[0] == "0":
                currentTimeStr = currentTime[1:]
            else:
                currentTimeStr = currentTime  # Fixed this :D
            output_folder = currentVideo + "_" + currentTimeStr
            videoPath = _DATA_DIR + currentVideo + "_" + currentTimeStr + "_orig.avi"
            if not os.path.exists(videoPath):
                print(videoPath + " not found")
                sys.exit(0)
            else:
                if not os.path.exists(_OUT_DIR + output_folder):
                    os.makedirs(_OUT_DIR + output_folder)
                    cmd = "ffmpeg -loglevel 0 -i " + videoPath + " -vf select='eq(n\,25)+eq(n\,35)+eq(n\,45)+eq(n\,55)+eq(n\,65)' -vsync 0 -s 224x224 " + _OUT_DIR + output_folder + "/frames%d.jpg"
                    os.system(cmd)
else:
    counter = 1
    for v in glob.glob(_DATA_DIR + "*"):
        print(v)
        counter += 1
        if not os.path.exists(v):
            print(v + " not found")
        else:
            output_folder = v.rsplit("/", 1)[1]
            output_folder = output_folder[:-4]
            # print(output_folder)
            if not os.path.exists(_OUT_DIR + output_folder):
                os.makedirs(_OUT_DIR + output_folder)
                cmd = "ffmpeg -loglevel 0 -i " + v + " -vf select='eq(n\,25)+eq(n\,35)+eq(n\,45)+eq(n\,55)+eq(n\,65)' -vsync 0 -s 224x224 " + _OUT_DIR + output_folder + "/frames%d.jpg"
                os.system(cmd)

# TODO The following code is for resizing frames

# type_resize = "rgb"
# set_type = "train"
# _DATA_DIR_RGB = "ava_" + set_type + "/rgb/"
# _DATA_DIR_FLOW_X = "/media/pedro/actv4/AVA-split/" + set_type + "/x/"
# _DATA_DIR_FLOW_Y = "/media/pedro/actv4/AVA-split/" + set_type + "/y/"
# _OUT_DIR_RGB = "ava_" + set_type + "_resized/rgb/"
# _OUT_DIR_FLOW_X = "/media/pedro/actv4/AVA-split/" + set_type + "_resize/x/"
# _OUT_DIR_FLOW_Y = "/media/pedro/actv4/AVA-split/" + set_type + "_resize/y/"
# output_res = (224, 224)
# if type_resize == "rgb":
#     fcount = 0
#    for dirs in os.listdir(_DATA_DIR_RGB):
#        print("rgb: " + str(fcount))
#        full_dir_path = _DATA_DIR_RGB + dirs + "/"
#        for filename in glob.glob(full_dir_path + "*.jpg"):
#            # print filename
#            frame = filename.rsplit("/", 1)[1]
#            savedir = _OUT_DIR_RGB + dirs + "/" + frame
#            if not os.path.exists(savedir):
#                # print frame
#                # Read image
#                img = Image.open(filename)
#                # Resize
#                # Methods are: Image.NEAREST, Image.BILINEAR, Image.BICUBIC,
#                # Image.LANCZOS
#                img = img.resize(output_res, Image.NEAREST)
#                # If output dir doesn't exist, create it
#                if not os.path.exists(_OUT_DIR_RGB + dirs + "/"):
#                    os.makedirs(_OUT_DIR_RGB + dirs + "/")
#                # Save img
#                img.save(savedir)
#        fcount += 1
