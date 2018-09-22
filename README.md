<!-- # Augmentation of Two Stream CNN architectures via attention, context or pose information -->
# Two Stream Multi-Label Attention Convolutional Neural Network with Context

We provide code for Two Stream Action Detection in Keras and each respective extension. *NOTE* Context only exists in the AVA dataset.
Generators to feed data are not used for AVA due to how temporal sampling is done. 

### Prerequisites

* python >= 2.7 or >= 3.6
* numpy
* ffmpeg built with gstreamer
* OpenCV 2.X or 3.X (build with opencv-contrib and ffmpeg and CUDA support)
* Pillow
* CUDA + cuDNN (we used 9 and 7.1, respectively)
* Tensorflow (for backend GPU) (`pip install tensorflow-gpu`)
* Keras >= 2.1.6 (`pip install keras`)
* pydot (`pip install pydot`)
* Pandas  and SciPy (because of dependencies)
* [imgaug](https://imgaug.readthedocs.io/en/latest/source/examples_basics.html) for augmentation (will use tensorflow tf.image API in the future)
* Maplotlib and Seaborn (`pip install seaborn`) for plots
* [gpu-flow](https://github.com/pedro-abreu/gpu_flow) if you want to extract TVL1 Optical Flow (requires OpenCV 2.X)
* [foveated-yolt](https://github.com/pedro-abreu/foveated_yolt) if you want to apply attention filters (requires python2)
* (Optional) Caffe 1.0 ([This](https://github.com/yjxiong/caffe) version if you wish to convert networks trained with yjxiong's version of Caffe (i.e TSN) to Keras). Note that we had to convert from this custom Caffe to Keras and as such, slightly modify the default InceptionV3 architecture to use pre-trained Kinetics weights (explained in the thesis). To understand the subtle (and low-level) differences in converting models from Caffe to Keras see [this](https://flyyufelix.github.io/2017/03/23/caffe-to-keras.html) great link.
* (Optional) OpenPose 1.3.0 if you wish to extract poses (most recent one as of writing, and thus Caffe (all dependencies) + OpenCV, can be the previous one)


### Data

* AVA -- For the AVA dataset due to computational power constraints and for quickly testing the architecture, we made our own split of the dataset called mini-AVA, you can download it [here](https://drive.google.com/open?id=1CfXJVxekmAtdxX6ng_j6Ed-XfBy6Bpk2)

<!---* AHA -- For the AHA dataset you can download our provided data it here: https://drive.google.com/drive/folders/11sfLyjtmtakF9kDzWEpAVwD5k4zjDkdV-->

<!-- * UCF101 -- For the UCF101 dataset you can download our provided data here (raw videos, flow (as rgb images), warped flow (as presented in the TSN paper), rgb (rescaled to 224x224) and pose (original and rescaled to 224x224 + joints)) [here](https://drive.google.com/open?id=16DXjG9J5YNQoXKPRaHaLU20MHcQijAs-). Note that the original UCF101 has flow provided in a non-efficient grayscale format.
-->
### Models

You can get our Keras models [here](https://drive.google.com/open?id=1HQT6bhJlRECFRdW6VngU03h_2yuiZJGY), includes pre-trained Kinetics, UCF101 models and our AVA models pre-trained on both.

### Extras

While not what was mainly tested we provide poses extracted from the full AVA dataset, heatmaps + json joints and a pose model (AlexNet). Openpose scripts to extract poses are used too (most recent OpenPose with face and hands information). If you want the UCF101 pose we can also provide it.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
