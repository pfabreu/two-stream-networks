import os

OUT_DIR = '/media/pedro/actv3/AHA/flow/'
DATA_DIR = "/media/pedro/actv3/AHA/videos/original/"
GPU_FLOW_DIR = '../../arch/flow_stream/gpu_flow/build/'


def _process_dataset():
    os.system(GPU_FLOW_DIR + "./compute_flow_si_warp --gpuID=0 --type=1 --vid_path=" +
              DATA_DIR + " --out_path=" + OUT_DIR)


def main():
    # Create directories for the classes
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR + "/")

    # Process dataset
    _process_dataset()


if __name__ == '__main__':
    main()
