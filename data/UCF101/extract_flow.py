# -*- coding: utf-8 -*-
# Go through every class folder, create an output class folder and run compute_flow for all those videos (do this 101 times)

import os
import glob

OUT_DIR = '/media/pedro/actv5/UCF101/flow_sf/'
DATA_DIR = '/media/pedro/actv3/UCF101/videos/'
GPU_FLOW_DIR = '../../arch/streams/gpu_flow/build/'

# Load video
vid_count = 0
class_folders = glob.glob(DATA_DIR + "*")
continue_idx = 0  # set to 0 if you want ot process all videos

for c in class_folders:
    cname = c.split("/")[-1]
    if not os.path.exists(OUT_DIR + cname):
        os.makedirs(OUT_DIR + cname)
    os.system(GPU_FLOW_DIR + "./compute_flow_si_warp --gpuID=0 --type=1 --vid_path=" +
              c + " --out_path=" + OUT_DIR + cname + " --skip=" + str(1))
