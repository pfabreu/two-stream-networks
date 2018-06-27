import csv
import os
import datetime
import glob

_DATA_DIR = "/media/pedro/actv4/AVA/pose/"
_OUTPUT_DIR = "/media/pedro/actv4/AVA/pose_segments"

# Load list of actions and separate them
actions = []
with open('files/ava_action_list_v2.1.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        actions.append(row)


# Load list of action snippets
set_type = "validation"
_OUTPUT_DIR = _OUTPUT_DIR + "_" + set_type + "/"


if set_type != "test":

    snippets_video = []
    snippets_time = []
    snippets_action = []

    if set_type == "validation":
        with open('files/ava_val_v2.1.csv') as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                snippets_video.append(row[0])
                snippets_time.append(row[1])
                snippets_action.append(row[6])

    elif set_type == "train":
        with open('files/ava_train_v2.1.csv') as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                snippets_video.append(row[0])
                snippets_time.append(row[1])
                snippets_action.append(row[6])

    snp_vid = ""
    snp_time = 0
    snips = []
    grouped_actions = []
    snp_vid = snippets_video[0]
    snp_time = snippets_time[0]
    grouped_actions.append(snippets_video[0])
    grouped_actions.append(snippets_time[0])
    for i in range(0, len(snippets_video)):
        if snp_vid == snippets_video[i] and snp_time == snippets_time[i]:
            grouped_actions.append(snippets_action[i])
        else:
            # print(grouped_actions)
            snips.append(grouped_actions)
            grouped_actions = []
            grouped_actions.append(snippets_video[i])
            grouped_actions.append(snippets_time[i])
            # TODO ADD THE FIRST ACTION!!
            grouped_actions.append(snippets_action[i])

        snp_vid = snippets_video[i]
        snp_time = snippets_time[i]

    somelist = [x for x in snips if not len(x) == 2]
    # somelist = [x for x in snips if len(x) == 3]

    DEBUG = True
    print len(somelist)
    for i in range(0, len(somelist)):
        video_name = somelist[i][0]
        timestamp = int(somelist[i][1])
        out_vid = _OUTPUT_DIR + video_name + \
            "_" + str(timestamp) + ".avi"
        # Process only videos you have not seen
        print i
        print out_vid
        if not os.path.exists(out_vid):
            print out_vid + "\t" + str(i) + "/" + str(len(somelist))

            timestamp_offset = 15 * 60
            if not os.path.exists(_DATA_DIR + video_name + ".avi"):
                print "panic: original video doesnt exist"
                print _DATA_DIR + video_name + ".avi"
                break
            timestamp = timestamp - timestamp_offset
            lower_bound = int(timestamp - 1.5)
            upper_bound = int(timestamp + 1.5)
            start_time = str(datetime.timedelta(seconds=lower_bound))
            end_time = str(datetime.timedelta(seconds=upper_bound))

            os.system("ffmpeg -loglevel panic -i " + _DATA_DIR + video_name +
                      ".avi" + " -ss " + start_time + " -to " + end_time + " " + out_vid)

    print "All segments processed!"
else:
    # Test segments
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
