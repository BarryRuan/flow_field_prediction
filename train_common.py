"""
Some training details defined in this file, such as placeholder, optimizer,
accuracy, error.
"""
import tensorflow as tf
from utils import get
import numpy as np


def regressor_placeholders(input_size=30, output_size=2, num_step=9):
    """
    Constructs the tensorflow placeholders needed as input to the network.
    
    Returns:
        two tensorflow placeholders. The first return value should be
        the placeholder for the input data. The second should be for the
        output data.
    """
    ffs = tf.placeholder(tf.float32, shape=[num_step, None, input_size], name='flow_fields')
    labels = tf.placeholder(tf.float32, shape=[None,output_size], name='labels')
    return ffs, labels


def unsupervised_optimizer(loss, lr=4e-4):
    """
    Constructs the training op needed to train the autoencoder model.

    Returns:
        the operation that begins the backpropogation through the network
        (i.e., the operation that minimizes the loss function).
    """
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    return train_op

def regressor_accuracy(labels, logits, partition='velocity'):
    """
    Constructs the accuracy metric given the ground truth labels and the
    network output logits.

    Returns:
        the accuracy value as a Tensorflow Tensor
    """

    # Norm of the difference of vectors in logits and labels 
    distance = tf.norm(logits-labels, axis=1)

    # For velocity field or magnitude map, a prediction is regarded as accuate only 
    # when the norm of deviation of the vector predicted by the model from its
    # true value is not greater than 2. For direction map, not greater than 0.25.
    if partition=='velocity' or partition=='magnitude':
        max_distance = tf.ones(shape=tf.shape(labels)[0:1])*2
    else:
        max_distance = tf.ones(shape=tf.shape(labels)[0:1])*0.25
    correct_prediction = tf.less(distance, max_distance)

    # accuracy rate = number of accurate vectors/total number of test vectors
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def mean_linear_error(images, reconstructed):
    """
    Constructs the mean liear error loss between the original images and the
    autoencoder reconstruction

    Returns:
        the mse loss as a Tensorflow Tensor
    """
    mle = tf.reduce_mean(tf.abs(tf.subtract(images,reconstructed)))
    return mle

def mean_squared_error(images, reconstructed):
    """
    Constructs the mean squared error loss between the original images and the
    autoencoder reconstruction

    Returns:
        the mse loss as a Tensorflow Tensor
    """
    mse = tf.losses.mean_squared_error(images,reconstructed)
    return mse

