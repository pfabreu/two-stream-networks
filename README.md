# Augmentation of Two Stream CNN architectures via attention, context or pose information

We provide code for Two Stream Action Recognition in Keras and each respective extension. *NOTE* Context only exists in the AVA dataset.
Generators to feed data are used for UCF101 but not for AVA.  Openpose scripts extract face and hand information and might be slower than expected for that reason

### Prerequisites

* python >= 2.7 or >= 3.6
* ffmpeg built with gstreamer
* OpenCV 2.X or 3.X (build with opencv-contrib and ffmpeg and CUDA support)
* CUDA + cuDNN (we used 9 and 7.1, respectively)
* Tensorflow (for backend GPU) (`pip install tensorflow-gpu`)
* [imgaug](https://imgaug.readthedocs.io/en/latest/source/examples_basics.html) for augmentation (will use tensorflow tf.image API in the future)
* Keras >= 2.1.6 (`pip install keras`)
* Pillow
* numpy
* Maplotlib and Seaborn (`pip install seaborn`) for plots
* Pandas  and SciPy (because of dependencies)
* [gpu-flow](https://github.com/pedro-abreu/gpu_flow) (requires OpenCV 2.X)
* [foveated-yolt](https://github.com/pedro-abreu/foveated_yolt) (requires python2)
* (Optional) OpenPose 1.3.0 if you wish to extract poses (most recent one as of writing, and thus Caffe (all dependencies) + OpenCV, can be the previous one)
* (Optional) Caffe 1.0 ([This](https://github.com/yjxiong/caffe) version if you wish to convert networks trained with yjxiong's version of Caffe (i.e TSN) to Keras). Note that we had to convert from this custom Caffe to Keras and as such, slightly modify the default InceptionV3 architecture to use pre-trained Kinetics weights (explained in the thesis). To understand the subtle (and low-level) differences in converting models from Caffe to Keras see [this](https://flyyufelix.github.io/2017/03/23/caffe-to-keras.html) great link.

### Data

* AVA -- For the AVA dataset due to computational power constraints and for quickly testing the architecture, we made our own split of the dataset called mini-AVA, you can download it [here](https://drive.google.com/open?id=1CfXJVxekmAtdxX6ng_j6Ed-XfBy6Bpk2)

<!---* AHA -- For the AHA dataset you can download our provided data it here: https://drive.google.com/drive/folders/11sfLyjtmtakF9kDzWEpAVwD5k4zjDkdV-->

* UCF101 -- For the UCF101 dataset you can download our provided data here (raw videos, flow (as rgb images), warped flow (as presented in the TSN paper), rgb (rescaled to 224x224) and pose (original and rescaled to 224x224 + joints)) [here](https://drive.google.com/open?id=16DXjG9J5YNQoXKPRaHaLU20MHcQijAs-). Note that the original UCF101 has flow provided in a non-efficient grayscale format.

### Models

You can get our Keras models [here](https://drive.google.com/open?id=1HQT6bhJlRECFRdW6VngU03h_2yuiZJGY), includes pure Kinetics, UCF101 models and AVA models pre-trained on both.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
