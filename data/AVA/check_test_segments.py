import glob
import cv2
import datetime
import os
from pymediainfo import MediaInfo
_DATA_DIR = "/media/pedro/actv4/AVA/snippets_test/"
_OUTPUT_DIR = "/media/pedro/actv4/AVA/segments_test/"
for v in glob.glob(_DATA_DIR + "*"):
    vname = v.split("/")[-1]
    path_splits = v.split("/")[:-1]
    output_name = vname[:-4]
    print vname
    for timestamp in range(902, 1799):
        out_vid = _OUTPUT_DIR + output_name + "_" + str(timestamp) + ".avi"
        fileInfo = MediaInfo.parse(out_vid)
        vid_streamcount = 0
        for track in fileInfo.tracks:
            if track.track_type == "Video":
                vid_streamcount += 1
        if vid_streamcount == 0:
            print out_vid
            break

print "All segments processed!"
