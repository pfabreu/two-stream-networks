#!/bin/sh
#python -O get_ava_performance.py \
#  -l ../ava_action_list_newsplit_v2.1_for_activitynet_2018.pbtxt.txt \
#  -g AVA_Test_Custom_Corrected.csv \
#  -d ../../../code_without_generators/output_test_rgb.csv \
#  -i 0.5

python -O get_ava_performance.py \
  -l ava_action_list_newsplit_v2.1_for_activitynet_2018.pbtxt.txt \
  -g AVA_Test_Custom_Corrected.csv \
  -i 0.5
