import tensorflow as tf
import numpy as np


def get_a_cell(lstm_size, forget_bias=1, keep_prob=0.5):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=forget_bias)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop

def multilayer_birnn(input, lstm_size=400, dim_out=2, forget_bias=0.7, keep_prob=0.5, num_layers=1, partition='velocity', use_auto=False):

    inputs = input

    for i in range(num_layers):

        with tf.variable_scope(None, default_name="bidirectional-rnn"):

            lstm_fw = get_a_cell(lstm_size,forget_bias=forget_bias, keep_prob=keep_prob)
            lstm_bw = get_a_cell(lstm_size,forget_bias=forget_bias, keep_prob=keep_prob)

            lstm_outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, inputs, dtype=tf.float32, time_major=True)

            inputs = tf.concat(lstm_outputs, 2)

            if i == num_layers-1:

                seq_output = tf.concat([lstm_outputs[0][4], lstm_outputs[1][5]], axis=1)

                if not use_auto:
                    with tf.variable_scope('softmax'):
                        softmax_w = tf.Variable(tf.truncated_normal([lstm_size*2, dim_out], stddev=0.1))
                        softmax_b = tf.Variable(tf.zeros(dim_out))
                    logits = tf.matmul(seq_output, softmax_w) + softmax_b
                else:
                    with tf.variable_scope('softmax'):
                        softmax_w = tf.Variable(tf.truncated_normal([lstm_size*2, lstm_size], stddev=0.1))
                        softmax_b = tf.Variable(tf.zeros(lstm_size))

                    logits = tf.matmul(seq_output, softmax_w) + softmax_b

                    with tf.variable_scope('softmax1'):
                        softmax_w1 = tf.Variable(tf.truncated_normal([lstm_size, dim_out], stddev=0.1))
                        softmax_b1 = tf.Variable(tf.zeros(dim_out))

                    logits = tf.matmul(logits, softmax_w1) + softmax_b1
    return logits 

def birnn(input, lstm_size=512, dim_out=2, forget_bias=0.7, keep_prob=0.5, partition='velocity', use_auto=False):

    with tf.name_scope('lstm'):
        lstm_fw = get_a_cell(lstm_size,forget_bias=forget_bias, keep_prob=keep_prob)
        lstm_bw = get_a_cell(lstm_size,forget_bias=forget_bias, keep_prob=keep_prob)

        lstm_outputs, final_state =tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, input, dtype=tf.float32, time_major=True)

        seq_output = tf.concat([lstm_outputs[0][4], lstm_outputs[1][5]], axis=1)

        if not use_auto:
            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([lstm_size*2, dim_out], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(dim_out))

            logits = tf.matmul(seq_output, softmax_w) + softmax_b
        else:
            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([lstm_size*2, lstm_size], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(lstm_size))

            logits = tf.matmul(seq_output, softmax_w) + softmax_b

            with tf.variable_scope('softmax1'):
                softmax_w1 = tf.Variable(tf.truncated_normal([lstm_size, dim_out], stddev=0.1))
                softmax_b1 = tf.Variable(tf.zeros(dim_out))

            logits = tf.matmul(logits, softmax_w1) + softmax_b1

        if partition=='direction':
            magnitude=tf.norm(logits,axis=1)
            magnitude=tf.stack([magnitude, magnitude], axis=1)
            logits=tf.div(logits,magnitude)
        return logits

