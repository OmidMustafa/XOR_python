"""Neural Network Implementation in Python equivalent to the one in C++.
Author: Arinol Team
Date: 27-Feb-2021
"""

# %% import the necessary libraries
# future library supports the standard library reorganization (PEP 3108) via one of several mechanisms,
# allowing most moved standard library modules to be accessed under their Python 3 names and locations in Python 2.
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

"""Random library Returns a random floating number between a specific range  i.e return a random number between 0 and 1."""
import random

"""Import Tensorflow library, TensorFlow is an end-to-end open source platform for machine learning. " \
It has a comprehensive, flexible ecosystem of tools, 
libraries and community resources that lets developers easily build and deploy ML powered applications."""
import tensorflow as tf


"""Import matplotlib.pyplot  which is a collection of functions that make matplotlib work like MATLAB. ' \
'Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, ' \
'plots some lines in a plotting area, decorates the plot with labels, etc."""
import matplotlib.pyplot as plt


"""Import numpy library that provides a multidimensional array object, various derived objects (such as masked arrays 
and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, 
shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical 
operations, random simulation and much more. """
import numpy as np

# %% 1- Generate the dataset for training and test sets (2 inputs and one output).

"""The dataset is made of two arrays the first one is a 2D array that consists of two columns and 500 rows, 
the first two columns hold inputs of boolean values, the second array is a 1D array that stores the XOR value of the
two inputs as an output result."""

# First input
a = 0

# Second input
b = 0

# Output
c = 0

# Create a 2D numpy array of zeros of shape (2 input columns) for the training input set.
train_data = np.array([[a, b]])

# Create a 1D numpy array of zeros of shape (1 output column) for training output set.
train_targets = np.array([c])

# Create a 2D numpy array of zeros of shape (2 input columns) for the test input set.
test_data = np.array([[a, b]])

# Create a 1D numpy array of zeros of shape (1 output column) for training output set.
test_targets = np.array([c])

"""Loop through 500 numbers for training starting from 1 with each iteration append a random values to train_data 
array respectively, after that append the XOR result of a and b to train_target array. """
for i in range(1, 500, 1):
    a = bool(random.getrandbits(1))
    b = bool(random.getrandbits(1))
    c = a ^ b
    train_data = np.append(train_data, [[a, b]], axis=0)
    train_targets = np.append(train_targets, [c])

"""Loop through 50 numbers for evaluation and testing starting from with each iteration 
append a,b values to test_data array respectively, after that append the XOR result value to test_target array. """
for i in range(1, 50, 1):
    a = bool(random.getrandbits(1))
    b = bool(random.getrandbits(1))
    c = a ^ b
    test_data = np.append(test_data, [[a, b]], axis=0)
    test_targets = np.append(test_targets, [c])

# %% 2- Building the model using keras library. keras is an API specification that describes how a Deep Learning
# framework should implement certain part, related to the model definition and training. Is framework agnostic and
# supports different backends (Theano, Tensorflow, ...) and tf.keras is the Tensorflow specific implementation of the
# Keras API specification. It adds the framework the support for many Tensorflow specific features like: perfect
# support for tf.data.Dataset as input objects, support for eager execution.'

# tf.keras.Sequential  provides training and inference features on the model.
model = tf.keras.Sequential([

    # tf.keras.layers.Flatten is used to flatten the input.For example,if flatten is applied to layer having input shape
    # as (batch_size, 2,2), then the output shape of the layer will be
    # (batch_size, 4), And then pass the training input size (input layer).
    tf.keras.layers.Flatten(input_shape=(2,)),

    # tf.keras.layers.Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation
    # is the element-wise activation function passed as the activation argument, kernel is a weights matrix created
    # by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True)."""

    # building the first hidden layer having 8 neurons with Tangent activation function.
    tf.keras.layers.Dense(8, activation='tanh'),

    # building the last layer that have one neuron.
    tf.keras.layers.Dense(1)
])

'''model.summary() , The summary can be created by calling the summary() function on the model that returns a string 
that in turn can be printed. The summary is textual and includes information about: 1 -The layers and their order in 
the model. 2 -The output shape of each layer. 3 -The number of parameters (weights) in each layer.4-The total number 
of parameters (weights) in the model. We can clearly see the output shape and number of weights in each layer. '''
print(model.summary())

# %% 3- Compile the model and define Loss and back-propagation method of the built model (compile the model).
'''model.compile, Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) 
such as Theano or TensorFlow. The backend automatically chooses the best way to represent the network for training 
and making predictions to run on your hardware, such as CPU or GPU or even distributed. When compiling, 
we must specify some additional properties required when training the network. Remember training a network means 
finding the best set of weights to map inputs to outputs in our dataset. We must specify the loss “Error” function to 
use to evaluate a set of weights, and the optimizer is used to search through different weights for the network and 
any optional metrics we would like to collect and report during training in another word the optimizer is the 
back-propagation. Like mentioned as part of the optimization algorithm, the error for the current state of the model 
must be estimated repeatedly. This requires the choice of an error function, conventionally called a loss function, 
that can be used to estimate the loss of the model so that the weights can be updated to reduce the loss on the next 
evaluation. '  loss='mse' , The Mean Squared Error, or MSE, loss is the default loss to use for regression problems. 
Mean squared error is calculated as the average of the squared differences between the predicted and actual values. 
The result is always positive regardless of the sign of the predicted and actual values and a perfect value is 0.0. 
The squaring means that larger mistakes result in more error than smaller mistakes, meaning that the model is 
punished for making larger mistakes. '''
model.compile(loss='mse',

              # optimizer='adam', We will define the optimizer as the efficient stochastic gradient descent algorithm
              # “adam“. Adam is an optimization algorithm that can be used instead of the classical stochastic
              # gradient descent procedure to update network weights iterative based in training data.This is a
              # popular version of gradient descent because it automatically tunes itself and gives good results in a
              # wide range of problems.

              optimizer='adam',

              # metrics=['mae'] A metric is a function that is used to judge the performance of the  model. Metrics
              # functions are similar to loss functions, except that the results from evaluating a metric are not
              # used when training the model. Note that we used Mean Abslout Error for evaluating the accuracy of the
              # our model by comparing it's performance on both Train & Test sets. We may use any loss function as a
              # metric.
              metrics=['mae'])

# %% 4- Train the model on the given training dataset.
'''model.fit, Fit Keras Model We have defined our model and compiled it ready for efficient computation. Now it is 
time to execute the model on some data. We can train or fit our model on our loaded data by calling the fit() function 
on the model. Training occurs over epochs and each epoch is split into batches. 1-	Epoch: One pass through all of the 
rows in the training dataset. 2-	Batch: One or more samples considered by the model within an epoch before weights 
are updated. 

One epoch is comprised of one or more batches, based on the chosen batch size and the model is fit for many epochs. '''
# The training process will run for a fixed number of iterations through the training dataset (train_data,
# train_targets) called epochs, that we must specify using the epochs argument. We must also set the number of
# dataset rows that are considered before the model weights are updated within each epoch, called the batch size and
# set using the batch_size argument. #we specify the epochs as 30 with a batch size 1,  These configurations was
# chosen experimentally by trial and error. We wanted to train the model enough so that it learns a good (or good
# enough) mapping of rows of input data to the output classification. The model will always have some error,
# but the amount of error will level out after some point for a given model configuration. This is called model
# convergence.

f = model.fit(train_data, train_targets, epochs=11, batch_size=1)

# %% 5- Evaluating the trained model on the test set of data to find it's loss and accuracy. after the training is
# completed.
'''model.evaluate, We have trained our neural network on the entire dataset and we can evaluate the performance of 
the network on the same dataset. This will only give us an idea of how well we have modelled the dataset (e.g. train 
accuracy), we have separated our data into train and test datasets for training and evaluation of the model. We 
evaluate the model on training dataset using the evaluate() function on model and pass it the same input and output 
used to train the model. This will generate a prediction for each input and output pair and collect scores, 
including the average loss and any metrics you have configured, such as accuracy. 

The evaluate() function will return a list with two values. The first will be the loss of the model on the dataset 
and the second will be the accuracy of the model on the dataset. We are only interested in reporting the loss and 
accuracy values. '''

# Print train and test loss and accuracy scores after training.
print("Train score:", model.evaluate(train_data, train_targets))
print("Test score:", model.evaluate(test_data, test_targets))

# %% 6- Plot the results of training and test accuracy and loss.
# The History object, When running the  model, Keras maintains a so-called History object in the background.
# This object keeps all loss values and other metric values in memory so that they can be used in e.g. TensorBoard,
# in Excel reports or indeed for our own custom visualizations, it’s the output of the fit operation. Hence,
# it can be accessed in the Python script.

# plot(f.history['loss'] and plot(f.history['mae'],  attribute is a dictionary recording training loss values and
# metrics values at successive epochs, as well as validation loss values and validation metrics values.
plt.plot(f.history['loss'], label='loss')
plt.plot(f.history['mae'], label='val_loss')

# %% 7- Make predictions.
# To receive valid and useful predictions, we must pre-process input for prediction in the same way that training data
# was pre-processed. For our trained model,we use the training we select a random sample from the evaluation
# data e.g 1,1. This data is in the form that was used to evaluate accuracy after each epoch of training,
# so it can be used to send test predictions without further pre-processing. np.array([[1, 1]], Create a 2D numpy
# array and fill it with two random booleans to test the prediction of our trained model.
T = np.array([[1, 1]])

# print out the array.  
print(T)

# np.rint() is a mathematical function round elements of the array to the nearest integer, and model.predict() predicts
# the addition of the two input integers, and to do so it expects the first parameter to be a numpy array which is our
# T array that contains the two integers.
a = np.rint(model.predict(T))

# print out result of the prediction.
print("Prediction: ", a)

# Save or load the model to be used later if needed.
# model.save('model.h5')
# model = tf.keras.models.load_model('model.h5')

# Enter the first bit.
print("Enter the first input:")
x = float(input())

# Enter the Second bit.
print("Enter the Second input:")
y = float(input())

# Explained, see above.
T = np.array([[x, y]])

# print out the array.
print(T)

# Explained see above.
a = model.predict(T)

# Explained see above.
print("Prediction: ", a)
