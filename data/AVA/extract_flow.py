import os

set_type = 'train'
OUT_DIR = '/media/pedro/actv4/ava-custom/flow_' + set_type + "/"
DATA_DIR = "/media/pedro/actv4/ava-split/split_segments_" + set_type
GPU_FLOW_DIR = '../../arch/streams/gpu_flow/build/'


def _process_dataset():

    os.system(GPU_FLOW_DIR + "./compute_flow_si_warp --gpuID=0 --type=1 --vid_path=" +
              DATA_DIR + " --out_path=" + OUT_DIR + " --skip=" + str(2))


def main():
    # Create directories for the classes
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR + "/")

    # Process dataset
    _process_dataset()


if __name__ == '__main__':
    main()
