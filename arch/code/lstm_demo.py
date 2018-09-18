from random import random
from numpy import array
from numpy import cumsum
import tensorflow as tf
import time
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional


def get_sequence(n_timesteps):
    # create a sequence of random numbers in [0,1]
    X = array([random() for _ in range(n_timesteps)])
    # calculate cut-off value to change class values
    limit = n_timesteps / 4.0
    # determine the class outcome for each item in cumulative sequence
    y = array([0 if x < limit else 1 for x in cumsum(X)])
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y

GPU = False
CPU = True
num_cores = 8

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                        device_count={'CPU': num_CPU, 'GPU': num_GPU})
session = tf.Session(config=config)
K.set_session(session)


# The classification problem has 1 sample (e.g. one sequence), a configurable number of timesteps, and one feature per timestep.
# We will define the sequences as having 10 timesteps.

# define problem properties
n_timesteps = 10
# Next, we can define an LSTM for the problem. The input layer will have 10 timesteps with 1 feature a piece, input_shape=(10, 1).
# define LSTM
model = Sequential()
# A sigmoid activation function is used on the output to predict the binary value.
# A TimeDistributed wrapper layer is used around the output layer so that one value per timestep can be predicted given the full sequence provided as input.
# This requires that the LSTM hidden layer returns a sequence of values (one per timestep) rather than a single value for the whole input sequence.
model.add(LSTM(20, input_shape=(10, 1), return_sequences=True))
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
#model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(1000):
    # generate new random sequence
    X, y = get_sequence(n_timesteps)
    # fit model for one epoch on this sequence
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate LSTM
X, y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
    print('Expected:', y[0, i], 'Predicted', yhat[0, i])
