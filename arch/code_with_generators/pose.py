# NOTE usage of generators in keras:
# The output of predict_generator() and predict() may not match if you are using a data generator created with flow_from_directory() from ImageDataGenerator(), even if your training data, model architecture, hyperparameters, and random seed are identical.
# If you are not aware of this, predict_generator() will likely give you garbage results.
# This is now harder to discover because some functions to deal with this from keras 1.0 are gone in keras 2.0.
# This also makes it harder to train multi-label models on large datasets.
