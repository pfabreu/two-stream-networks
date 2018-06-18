import glob
import datetime
import os

_DATA_DIR = "/media/pedro/actv4/AVA/snippets_test/"
_OUTPUT_DIR = "/media/pedro/actv4/AVA/segments_test/"
for v in glob.glob(_DATA_DIR + "*"):
    vname = v.split("/")[-1]
    path_splits = v.split("/")[:-1]
    output_name = vname[:-4]
    for timestamp in range(902, 1799):
        out_vid = _OUTPUT_DIR + output_name + \
            "_" + str(timestamp) + ".avi"
        print out_vid
        if not os.path.exists(out_vid):
            print out_vid
            print timestamp
            timestamp_offset = 15 * 60
            timestamp = timestamp - timestamp_offset
            lower_bound = int(timestamp - 1.5)
            upper_bound = int(timestamp + 1.5)
            start_time = str(datetime.timedelta(seconds=lower_bound))
            end_time = str(datetime.timedelta(seconds=upper_bound))

            os.system("ffmpeg -loglevel panic -i " + _DATA_DIR + vname + " -ss " + start_time + " -to " + end_time + " " + out_vid)

print "All segments processed!"
