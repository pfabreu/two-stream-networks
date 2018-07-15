r"""Compute action detection performance for the AVA dataset.

Please send any questions about this code to the Google Group ava-dataset-users:
https://groups.google.com/forum/#!forum/ava-dataset-users

Example usage:
python -O get_ava_performance.py \
  -l ava/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt \
  -g ava_val_v2.1.csv \
  -e ava_val_excluded_timestamps_v2.1.csv \
  -d your_results.csv
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import defaultdict
import csv
import logging
import pprint
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ava import object_detection_evaluation
from ava import standard_fields


def print_time(message, start):
    logging.info("==> %g seconds to %s", time.time() - start, message)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def read_csv(csv_file, class_whitelist=None):
    """Loads boxes and class labels from a CSV file in the AVA format.

    CSV file format described at https://research.google.com/ava/download.html.

    Args:
      csv_file: A file object.
      class_whitelist: If provided, boxes corresponding to (integer) class labels
        not in this set are skipped.

    Returns:
      boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
      labels: A dictionary mapping each unique image key (string) to a list of
        integer class lables, matching the corresponding box in `boxes`.
      scores: A dictionary mapping each unique image key (string) to a list of
        score values lables, matching the corresponding label in `labels`. If
        scores are not provided in the csv, then they will default to 1.0.
    """
    start = time.time()
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row) in [7, 8], "Wrong number of columns: " + row
        image_key = make_image_key(row[0], row[1])
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        action_id = int(row[6])
        if class_whitelist and action_id not in class_whitelist:
            continue
        score = 1.0
        if len(row) == 8:
            score = float(row[7])
        boxes[image_key].append([y1, x1, y2, x2])
        labels[image_key].append(action_id)
        scores[image_key].append(score)
    print_time("read file " + csv_file.name, start)
    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.

    Args:
      exclusions_file: A file object containing a csv of video-id,timestamp.

    Returns:
      A set of strings containing excluded image keys, e.g. "aaaaaaaaaaa,0904",
      or an empty set if exclusions file is None.
    """
    excluded = set()
    if exclusions_file:
        reader = csv.reader(exclusions_file)
        for row in reader:
            assert len(row) == 2, "Expected only 2 columns, got: " + row
            excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
      labelmap_file: A file object containing a label map protocol buffer.

    Returns:
      labelmap: The label map in the form used by the object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
      class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    for line in labelmap_file:
        if line.startswith("  name:"):
            name = line.split('"')[1]
        elif line.startswith("  id:") or line.startswith("  label_id:"):
            class_id = int(line.strip().split(" ")[-1])
            labelmap.append({"id": class_id, "name": name})
            class_ids.add(class_id)
    return labelmap, class_ids


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]


def split_interleave(A):
    lists = split_list(A, wanted_parts=4)
    D = [val for tup in zip(*lists) for val in tup]
    return D


def run_evaluation(labelmap, groundtruth, exclusions, iou):

    # Make sure not to mess this up
    filters = []
    filters.append("rgb")
    filters.append("crop")
    filters.append("gauss")
    filters.append("fovea")
    all_detections = []
    # Flow, context, 2-stream, 3-stream run

    # RGB run
    all_detections.append(open("../../../code_without_generators/output_test_rgb.csv", 'rb'))
    all_detections.append(open("../../../code_without_generators/output_test_crop.csv", 'rb'))
    all_detections.append(open("../../../code_without_generators/output_test_gauss.csv", 'rb'))
    all_detections.append(open("../../../code_without_generators/output_test_fovea.csv", 'rb'))

    all_gndtruths = []
    # TODO Fix this dirty hack lol
    all_gndtruths.append(open("AVA_Test_Custom_Corrected.csv", 'rb'))
    all_gndtruths.append(open("AVA_Test_Custom_Corrected.csv", 'rb'))
    all_gndtruths.append(open("AVA_Test_Custom_Corrected.csv", 'rb'))
    all_gndtruths.append(open("AVA_Test_Custom_Corrected.csv", 'rb'))
    """Runs evaluations given input files.

    Args:
      labelmap: file object containing map of labels to consider, in pbtxt format
      groundtruth: file object
      detections: file object
      exclusions: file object or None.
    """
    categories, class_whitelist = read_labelmap(labelmap)
    logging.info("CATEGORIES (%d):\n%s", len(categories),
                 pprint.pformat(categories, indent=2))
    excluded_keys = read_exclusions(exclusions)

    # Reads detections data.
    x_axis = []
    xpose_ax = []
    xobj_ax = []
    xhuman_ax = []
    ypose_ax = []
    yobj_ax = []
    yhuman_ax = []
    colors_pose = []
    colors_obj = []
    colors_human = []
    finalmAPs = []
    colors = []
    for detections, gndtruth, filter_type in zip(all_detections, all_gndtruths, filters):
        pascal_evaluator = None
        metrics = None
        actions = None
        start = 0

        pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
            categories, matching_iou_threshold=iou)

        # Reads the ground truth data.
        boxes, labels, _ = read_csv(gndtruth, class_whitelist)
        start = time.time()
        for image_key in boxes:
            if image_key in excluded_keys:
                logging.info(("Found excluded timestamp in ground truth: %s. "
                              "It will be ignored."), image_key)
                continue
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    standard_fields.InputDataFields.groundtruth_boxes:
                        np.array(boxes[image_key], dtype=float),
                    standard_fields.InputDataFields.groundtruth_classes:
                        np.array(labels[image_key], dtype=int),
                    standard_fields.InputDataFields.groundtruth_difficult:
                        np.zeros(len(boxes[image_key]), dtype=bool)
                })
        print_time("convert groundtruth", start)

        # Run evaluation
        boxes, labels, scores = read_csv(detections, class_whitelist)
        start = time.time()
        for image_key in boxes:
            if image_key in excluded_keys:
                logging.info(("Found excluded timestamp in detections: %s. "
                              "It will be ignored."), image_key)
                continue
            pascal_evaluator.add_single_detected_image_info(
                image_key, {
                    standard_fields.DetectionResultFields.detection_boxes:
                        np.array(boxes[image_key], dtype=float),
                    standard_fields.DetectionResultFields.detection_classes:
                        np.array(labels[image_key], dtype=int),
                    standard_fields.DetectionResultFields.detection_scores:
                        np.array(scores[image_key], dtype=float)
                })
        print_time("convert detections", start)

        start = time.time()
        metrics = pascal_evaluator.evaluate()
        print_time("run_evaluator", start)

        # TODO Show a pretty histogram here besides pprint
        actions = list(metrics.keys())

        final_value = 0.0
        for m in actions:
            ms = m.split("/")[-1]

            if ms == 'mAP@' + str(iou) + 'IOU':
                final_value = metrics[m]
                finalmAPs.append(final_value)
            else:
                # x_axis.append(ms)
                # y_axis.append(metrics[m])
                for cat in categories:
                    if cat['name'].split("/")[-1] == ms:
                        if cat['id'] <= 9:
                            xpose_ax.append("(" + filter_type + ") " + ms)
                            ypose_ax.append(metrics[m])
                            colors_pose.append('red')
                        elif cat['id'] <= 22:
                            xobj_ax.append("(" + filter_type + ") " + ms)
                            yobj_ax.append(metrics[m])
                            colors_obj.append('blue')
                        else:
                            xhuman_ax.append("(" + filter_type + ") " + ms)
                            yhuman_ax.append(metrics[m])
                            colors_human.append('green')
        pascal_evaluator = None

    x_axis = split_interleave(xpose_ax) + split_interleave(xobj_ax) + split_interleave(xhuman_ax)
    y_axis = split_interleave(ypose_ax) + split_interleave(yobj_ax) + split_interleave(yhuman_ax)
    colors = colors_pose + colors_obj + colors_human

    plt.ylabel('frame-mAP')
    top = 0.6
    sns.set_style("whitegrid")
    g = sns.barplot(x_axis, y_axis, palette=colors)
    ax = g
    # annotate axis = seaborn axis
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='gray', rotation=90, xytext=(0, 20),
                    textcoords='offset points')
    _ = g.set_ylim(0, top)  # To make space for the annotations
    pprint.pprint(metrics, indent=2)
    plt.xticks(rotation=-90)
    title = ""
    for filter_type, mAP in zip(filters, finalmAPs):
        ft = filter_type + ': mAP@' + str(iou) + 'IOU = ' + str(mAP) + '\n'
        title += ft
    plt.title(title)
    plt.show()

    sys.exit(0)

    # Confusion matrix
    classes = []
    for i in categories:
        classes.append(i['name'])
    cm = confusion_matrix(groundtruth, detections, x_axis)
    g = sns.heatmap(cm, annot=True, fmt="d", cmap=sns.cubehelix_palette(8), xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor='black', cbar=False)

    i = 0
    for ytick_label, xtick_label in zip(g.axes.get_yticklabels(), g.axes.get_xticklabels()):
        if i <= 9:
            ytick_label.set_color("r")
            xtick_label.set_color("r")

        elif i <= 22:
            ytick_label.set_color("b")
            xtick_label.set_color("b")
        else:
            ytick_label.set_color("g")
            xtick_label.set_color("g")
        i += 1
    plt.xticks(rotation=-90)
    plt.title("NOTE: This only works if we are doing pure classification (i.e using gnd truth BBs)")
    plt.show()


def confusion_matrix(groundtruth, detections, x_axis):
    cm = np.zeros([len(x_axis), len(x_axis)], np.int32)

    gnd_dict = {}
    det_dict = {}

    # print(groundtruth)
    # print(detections)

    # Load gndtruth
    groundtruth.seek(0)
    reader = csv.reader(groundtruth)
    #print("Parsing file")

    for row in reader:
        video = row[0]
        kf = row[1]
        bbs = str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
        i = video + "@" + kf.lstrip("0") + "@" + bbs
        gnd_dict[i] = []
    groundtruth.seek(0)
    for row in reader:
        video = row[0]
        kf = row[1]
        bbs = str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
        i = video + "@" + kf.lstrip("0") + "@" + bbs
        gnd_dict[i].append(row[6])

    # Load predictions
    detections.seek(0)
    reader = csv.reader(detections)
    for row in reader:
        video = row[0]
        kf = row[1]
        bbs = str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
        i = video + "@" + kf.lstrip("0") + "@" + bbs
        det_dict[i] = []
    detections.seek(0)
    for row in reader:
        video = row[0]
        kf = row[1]
        bbs = str(row[2]) + "@" + str(row[3]) + "@" + str(row[4]) + "@" + str(row[5])
        i = video + "@" + kf.lstrip("0") + "@" + bbs
        det_dict[i].append(row[6])

    # TODO For softmax actions normal count
    for key, gnd_acts in gnd_dict.items():
        #print("KEY: " + key)
        det_acts = det_dict[key]
        # print(gnd_acts)
        # print(det_acts)
        gnd_pose = -1
        det_pose = -1
        for a in gnd_acts:
            if int(a) <= 10:
                # print(a)
                gnd_pose = int(a) - 1
        for a in det_acts:
            if int(a) <= 10:
                det_pose = int(a) - 1
        if gnd_pose != -1 and det_pose != -1:
            cm[gnd_pose, det_pose] += 1
            cm[det_pose, gnd_pose] += 1
    # TODO For the other two, if there is a correct predicted action count it, if there is an incorrect prediction either count it as None (if there was no action)
    # or add 1 to all the other correct actions
    return cm


def parse_arguments():
    """Parses command-line flags.

    Returns:
      args: a named tuple containing three file objects args.labelmap,
      args.groundtruth, and args.detections.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--labelmap",
        help="Filename of label map",
        type=argparse.FileType("r"),
        default="../ava_action_list_newsplit_v2.1_for_activitynet_2018.pbtxt.txt")
    parser.add_argument(
        "-g",
        "--groundtruth",
        help="CSV file containing ground truth.",
        type=argparse.FileType("rb"),
        required=True)
    parser.add_argument(
        "-e",
        "--exclusions",
        help=("Optional CSV file containing videoid,timestamp pairs to exclude "
              "from evaluation."),
        type=argparse.FileType("r"),
        required=False)
    parser.add_argument(
        "-i",
        "--iou",
        help="Optional IoU value ",
        type=float,
        required=False)

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    print(args)
    run_evaluation(**vars(args))


if __name__ == "__main__":
    main()
