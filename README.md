# Three Stream Action Recognition

We also provide code for Two Stream Action Recognition in Keras for the AVA and UCF101 datasets.

### Prerequisites

* python >= 2.7 or >= 3.6
* OpenCV 2.X or 3.X built with ffmpeg and gstreamer
* CUDA + cuDNN (we used 9 and 7, respectively)
* OpenPose (and thus Caffe (all dependencies) + OpenCV, can be the previous one)
* Tensorflow (for backend) (`pip install tensorflow-gpu`)
* Keras >= 2.1.6 (`pip install keras`)
* Pillow
* numpy

### Custom Libraries

* [gpu-flow]() (requires OpenCV 2.X)
* [foveated-yolt]() (requires python2)

### Data

For the AVA dataset due to computational power constraints and for quickly testing the architecture, we made our own split of the dataset called mini-AVA, you can download it here:

For the AHA dataset you can download our provided data it here: https://drive.google.com/drive/folders/11sfLyjtmtakF9kDzWEpAVwD5k4zjDkdV

For the UCF101 dataset you can download our provided data here (only flow (as rgb images), rgb (rescaled to 224x224) and pose (original and rescaled to 224x224)):

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
