#!/bin/sh

# Old threshold code
# python -O get_ava_performance_threshold.py \
#  -l ava_action_list_newsplit_v2.1_for_activitynet_2018.pbtxt.txt \
#  -g AVA_Val_Custom_Corrected.csv \
#  -i 0.5

python -O get_ava_performance_bettergraphs.py \
  -l ava_action_list_newsplit_v2.1_for_activitynet_2018.pbtxt.txt \
  -g ../../../data/AVA/files/AVA_Test_Custom_Corrected.csv \
  -i 0.5
