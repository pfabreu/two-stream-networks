# Extension of Two Stream CNN architectures via context, pose, facial or hand info

We provide code for Two Stream Action Recognition in Keras and each respective extension. *NOTE* Context only exists in the AVA dataset.
Generators to feed data are used for UCF101 and for AHA but not for AVA.

### Prerequisites

* python >= 2.7 or >= 3.6
* OpenCV 2.X or 3.X built with ffmpeg and gstreamer
* CUDA + cuDNN (we used 9 and 7, respectively)
* OpenPose (and thus Caffe (all dependencies) + OpenCV, can be the previous one)
* Tensorflow (for backend) (`pip install tensorflow-gpu`)
* Keras >= 2.1.6 (`pip install keras`)
* Pillow
* numpy
* [gpu-flow](https://github.com/pedro-abreu/gpu_flow) (requires OpenCV 2.X)
* [foveated-yolt](https://github.com/pedro-abreu/foveated_yolt) (requires python2)

### Data

* AVA -- For the AVA dataset due to computational power constraints and for quickly testing the architecture, we made our own split of the dataset called mini-AVA, you can download it here: https://drive.google.com/open?id=1CfXJVxekmAtdxX6ng_j6Ed-XfBy6Bpk2

* AHA -- For the AHA dataset you can download our provided data it here: https://drive.google.com/drive/folders/11sfLyjtmtakF9kDzWEpAVwD5k4zjDkdV

* UCF101 -- For the UCF101 dataset you can download our provided data here (only flow (as rgb images), rgb (rescaled to 224x224) and pose (original and rescaled to 224x224)): https://drive.google.com/open?id=16DXjG9J5YNQoXKPRaHaLU20MHcQijAs-

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
