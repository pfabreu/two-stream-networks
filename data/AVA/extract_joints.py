import glob
import os
import json
import numpy as np
import pprint
set_type = "val"
_INPUT_DIR = "/media/pedro/actv-ssd/pose_noface_" + set_type + "_joints"
_OUTPUT_DIR = _INPUT_DIR + "_npy/"

for v in glob.glob(_INPUT_DIR + "/*"):
    poses = []
    file = v.split("/")[-1] + ".npy"
    print file
    for f in glob.glob(v + "/*.json"):
        # print(f)
        frame_count = f.split("/")[-1]
        frame_count = frame_count.split("_")[2]
        frame_count = int(frame_count)
        # print frame_count

        # Process all files into numpy array
        with open(f) as json_f:
            data = json.load(json_f)
            # print data
            frame_poses = []
            # With data, you can now also find values like so:
            ppl = data["people"]
            # print ppl
            p_count = 0
            for person in ppl:
                if person[]:
                if person[]:
                if person[]:

                jts = person['pose_keypoints_2d']
                jts = [jts[n:n + 3] for n in range(0, len(jts), 3)]
                jts.insert(0, p_count)
                jts.insert(0, frame_count)
                jts.insert(0, [frame_count])
                # sprint jts
                poses.append(jts)
                p_count += 1
            # frame_count += 1
        # Save numpy array for this segment
    poses = np.array(poses)
    if poses != []:
        poses = poses[poses[:, 1].argsort()]
    # print poses[0, 1:]
    np.save(_OUTPUT_DIR + file, poses)
