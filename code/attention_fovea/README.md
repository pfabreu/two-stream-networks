# Fovea, Gaussian and Crop Filters

This repo has fast filters for eliptic foveation, gaussian blur and cropping for video segments.
It also extracts frames from the videos and writes them in folders (easy to adapt scripts if you want video output). Works with OpenCV 3 or OpenCV 2.


To compile from root: 

```
mkdir build && cd build && cmake ..

make -j4
```

To run each script runs a filter:
```
python crop.py
python gauss.py
python src/python_bindings/foveateAVA.py
```
