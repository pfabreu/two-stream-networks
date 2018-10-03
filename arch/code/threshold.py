import numpy as np
import utils
from rgb_data import get_AVA_set
import voting
import pickle

# File with predictions
stream = "fusion"
filter_type = "avg_fovea"
filename = "thresholds/context_fusion/predictions_" + stream + "_" + filter_type + "_1809281055.pickle"
with open(filename, 'rb') as handle:
    predictions = pickle.load(handle)


root_dir = '../../data/AVA/files/'

# Load groundtruth (test or val set)
# Load list of action classes and separate them (from utils_stream)
classes = utils.get_AVA_classes(root_dir + 'ava_action_list_custom.csv')
partition = get_AVA_set(classes=classes, filename=root_dir + "AVA_Val_Custom_Corrected.csv", soft_sigmoid=True)
test_splits = utils.make_chunks(original_list=partition, size=len(partition), chunk_size=2**11)

# Voting
thresh_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for thresh in thresh_values:
    print("Testing threshold " + str(thresh) + " ")
    pose_votes = {}
    obj_votes = {}
    human_votes = {}

    for row in partition:
        row = row.split("@")
        i = row[0] + "@" + row[1] + "@" + str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
        pose_votes[i] = np.zeros(utils.POSE_CLASSES)
        obj_votes[i] = np.zeros(utils.OBJ_HUMAN_CLASSES)
        human_votes[i] = np.zeros(utils.HUMAN_HUMAN_CLASSES)

    result_csv = filename.split(".")[0] + "_" + str(thresh) + ".csv"
    pred_chunk = 0
    for testIDS in test_splits:

        voting.pred2classes(testIDS, predictions[pred_chunk], pose_votes, obj_votes, human_votes, thresh=thresh)
        pred_chunk += 1

    with open(result_csv, "a") as output_file:
        for key in pose_votes:
            idx = key.split("@")
            actions = []
            pv = pose_votes[key]
            pose_vote = pv.argmax(axis=0) + 1
            actions.append(pose_vote)

            # Get 3 top voted object
            ov = obj_votes[key]
            top_three_obj_votes = ov.argsort()[-3:][::-1] + utils.POSE_CLASSES + 1
            for t in top_three_obj_votes:
                if t != 0:  # Often there might only be two top voted or one
                    actions.append(t)
            # Get 3 top voted human
            hv = human_votes[key]
            top_three_human_votes = hv.argsort()[-3:][::-1] + utils.POSE_CLASSES + utils.OBJ_HUMAN_CLASSES + 1
            for t in top_three_human_votes:
                if t != 0:  # Often there might only be two top voted or one
                    actions.append(t)

            video_name = idx[0]
            timestamp = idx[1]
            bb_topx = idx[2]
            bb_topy = idx[3]
            bb_botx = idx[4]
            bb_boty = idx[5]
            for a in actions:
                line = video_name + "," + timestamp + "," + bb_topx + "," + bb_topy + "," + bb_botx + "," + bb_boty + "," + str(a)
                output_file.write("%s\n" % line)
