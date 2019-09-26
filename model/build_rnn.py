'''
Build RNN 
    Constructs a tensorflow graph for a convolutional neural network
    Usage: from model.build_rnn import rnn 
    This model is mainly used for velocity map
'''
import numpy as np
import tensorflow as tf

# Create a single cell 
def get_a_cell(lstm_size, forget_bias=1, keep_prob=0.5):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=forget_bias)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop

def rnn(input, lstm_size=400, dim_out=2, forget_bias=0.7, keep_prob=0.5, num_layers=3, partition='velocity', use_auto=False):

    with tf.name_scope('lstm'):

        cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(lstm_size,forget_bias=forget_bias, keep_prob=keep_prob) for _ in range(num_layers)]
        )
    
        # Expande time sequence through dynamic_rnn
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32, time_major=True)

        seq_output = lstm_outputs[-1]

        if not use_auto:
            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([lstm_size, dim_out], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(dim_out))

            logits = tf.matmul(seq_output, softmax_w) + softmax_b
        else:
            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([lstm_size, lstm_size//2], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(lstm_size/2))

            logits = tf.matmul(seq_output, softmax_w) + softmax_b

            with tf.variable_scope('softmax1'):
                softmax_w1 = tf.Variable(tf.truncated_normal([lstm_size//2, dim_out], stddev=0.1))
                softmax_b1 = tf.Variable(tf.zeros(dim_out))

            logits = tf.matmul(logits, softmax_w1) + softmax_b1

        if partition=='direction':
            magnitude=tf.norm(logits,axis=1)
            magnitude=tf.stack([magnitude, magnitude], axis=1)
            logits=tf.div(logits,magnitude)
        return logits

