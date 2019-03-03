# Two Stream Attention Convolutional Neural Networks

We provide code for Two Stream Action Detection in tf.keras and each respective extension.

### Prerequisites

* python >= 2.7
* numpy (`pip install numpy`)
* ffmpeg built with gstreamer (`sudo apt install ffmpeg`)
* OpenCV 2.X or 3.X (build with opencv-contrib and ffmpeg and CUDA support)
* Pillow (`pip install Pillow`)
* CUDA + cuDNN (we used 9 and 7.1, respectively)
* Tensorflow (for backend GPU) (`pip install tensorflow-gpu`)
* Keras >= 2.1.6 (`pip install keras`)
* Pandas (`pip install pandas`) and SciPy (`pip install scipy`) (because of dependencies)
* Maplotlib + Seaborn (`pip install seaborn`) for plots

### Data
* AVA -- For the AVA dataset due to computational power constraints and for quickly testing the architecture, we made our own split of the dataset called mini-AVA, you can download it [here](https://www.dropbox.com/sh/vz2fgkucq40w9yo/AABpDyiViYSW4D1BGmNal6Mma?dl=0)

<!-- * UCF101 -- For the UCF101 dataset you can download our provided data here (raw videos, flow (as rgb images), warped flow (as presented in the TSN paper), rgb (rescaled to 224x224) and pose (original and rescaled to 224x224 + joints)) [here](https://drive.google.com/open?id=16DXjG9J5YNQoXKPRaHaLU20MHcQijAs-). Note that the original UCF101 has flow provided in a non-efficient grayscale format.
-->

### Models

You can get our Keras models [here](https://www.dropbox.com/sh/jlg3qzxw7kr2lvx/AADIxcp_XaW4I5xROFYG4u3ba?dl=0), includes pre-trained Kinetics, UCF101 models and our AVA models pre-trained on both.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
