import csv
import glob
import os
import json
import cv2
import numpy as np
import datetime

#_DATA_DIR = "/media/pedro/actv4/AVA/pose/"
#_OUTPUT_DIR = "/media/pedro/actv4/AVA/pose_segments/segments"
_DATA_DIR = "test/"
_OUTPUT_DIR = "test_output"
# Load list of actions and separate them
actions = []
with open('ava_action_list_v2.1.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        actions.append(row)


# Load list of action snippets
set_type = "train"
_OUTPUT_DIR = _OUTPUT_DIR + "_" + set_type + "/"

if not os.path.exists(_OUTPUT_DIR + "joints"):
    os.makedirs(_OUTPUT_DIR + "joints")

snippets_video = []
snippets_time = []
snippets_action = []

if set_type == "validation":
    with open('ava_val_v2.1.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            snippets_video.append(row[0])
            snippets_time.append(row[1])
            snippets_action.append(row[6])

elif set_type == "train":
    with open('ava_train_v2.1.csv') as csvDataFile:
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
        snips.append(grouped_actions)
        grouped_actions = []
        grouped_actions.append(snippets_video[i])
        grouped_actions.append(snippets_time[i])
        grouped_actions.append(snippets_action[i])

    snp_vid = snippets_video[i]
    snp_time = snippets_time[i]

somelist = [x for x in snips if not len(x) == 2]

for i in range(0, len(somelist)):
    video_name = somelist[i][0]
    timestamp = int(somelist[i][1])
    out_vid = _OUTPUT_DIR + video_name + \
        "_" + str(timestamp) + ".avi"
    # Process only videos you have not seen
    print i
    if not os.path.exists(out_vid):
        print out_vid + "\t" + str(i) + "/" + str(len(somelist))

        timestamp_offset = 15 * 60
        if not os.path.exists(_DATA_DIR + video_name + ".avi"):
            print "panic: original video doesnt exist"
            print _DATA_DIR + video_name + ".avi"
            continue # TODO (should be break) only do this for now that we are testing this
            #break
        timestamp = timestamp - timestamp_offset
        lower_bound = int(timestamp - 1.5)
        upper_bound = int(timestamp + 1.5)
        start_time = str(datetime.timedelta(seconds=lower_bound))
        end_time = str(datetime.timedelta(seconds=upper_bound))

        os.system("ffmpeg -loglevel panic -i " + _DATA_DIR + video_name +
                  ".avi" + " -ss " + start_time + " -to " + end_time + " " + out_vid)

        # Load all json joints in AVA/pose/joints
        # Output all joints in a single friendlier .npy format per segment in
        # AVA/pose/joints_npy
        # Load original video and get it's fps
        if os.path.exists("test_Videos/"+video_name+".mkv"):
            original_vid = cv2.VideoCapture("test_Videos/"+video_name+".mkv")
        elif os.path.exists("test_Videos/"+video_name+".mp4"):
            original_vid = cv2.VideoCapture("test_Videos/"+video_name+".mp4")
        elif os.path.exists("test_Videos/"+video_name+".webm"):
            original_vid = cv2.VideoCapture("test_Videos/"+video_name+".webm")
        else:
            print "Couldn't get FPS from original video!"
            break
        fps = original_vid.get(cv2.CAP_PROP_FPS)
        lb = int(lower_bound * fps)
        lower_bound_keypoint = _DATA_DIR + "joints/" + video_name + "/" + video_name + "_" + str('{:012}'.format(lb)) + "_keypoints.json"
        print "LB: " + lower_bound_keypoint
        ub = int(upper_bound * fps)
        upper_bound_keypoint = _DATA_DIR + "joints/" + video_name + "/" + video_name + "_" + str('{:012}'.format(ub)) + "_keypoints.json"
        print "UB: " + upper_bound_keypoint
        json_joint_path = _DATA_DIR + "joints/" + video_name
        poses = []
        frame_count = 0
        for f in glob.glob(json_joint_path + "/*.json"):
            # Go through all json files that represent
            with open(f) as json_f:
                if f >= lower_bound_keypoint and f <= upper_bound_keypoint:
                    print f
                    # This is a single frame
                    data = json.load(json_f)
                    #pprint(data)
                    # Create empy numpy array for this frame which will have size #nppl x #joints
                    frame_poses = []
                    # With data, you can now also find values like so:
                    ppl = data["people"]
                    # Result for COCO (18 body parts)
                    # POSE_COCO_BODY_PARTS {
                    #     {0,  "Nose"},
                    #     {1,  "Neck"},
                    #     {2,  "RShoulder"},
                    #     {3,  "RElbow"},
                    #     {4,  "RWrist"},
                    #     {5,  "LShoulder"},
                    #     {6,  "LElbow"},
                    #     {7,  "LWrist"},
                    #     {8,  "RHip"},
                    #     {9,  "RKnee"},
                    #     {10, "RAnkle"},
                    #     {11, "LHip"},
                    #     {12, "LKnee"},
                    #     {13, "LAnkle"},
                    #     {14, "REye"},
                    #     {15, "LEye"},
                    #     {16, "REar"},
                    #     {17, "LEar"},
                    #     {18, "Background"},
                    # }
                    p_count = 0
                    for person in ppl:
                        jts = person['pose_keypoints_2d']
                        jts = [jts[n:n+3] for n in range(0, len(jts), 3)]
                        jts.insert(0, [p_count])
                        jts.insert(0, [frame_count])
                        poses.append(jts)
                        p_count += 1
                    frame_count+=1
        poses = np.array(poses)
        print poses.shape
        

print "All pose segments processed!"
