import argparse
import tensorflow as tf
import numpy as np
import utils
import random
from data.RNNDataSet import RNNDataSet 
from model.build_birnn import birnn
from model.build_rnn import rnn
from train_common import *
from utils import get

def pred_rnn(sess, saver, save_path, images, logits, food, ff_type):
    """
    Trains a tensorflow model of an autoencoder to create sparse
    embeddings of an image dataset. Periodically saves model checkpoints
    and reports the network performance on a validation set.
    """
    if ff_type=='magnitude':
        model_pred = np.zeros(1).reshape(1,1)
    else:
        model_pred = np.zeros(2).reshape(1,2)
    while not food.finished_test_epoch():
        batch_images = food.get_batch(partition='test')
        feed_dict = {images: batch_images}
        batch_pred = sess.run(logits, feed_dict=feed_dict)
        model_pred = np.concatenate([model_pred, batch_pred])
    return model_pred[1:]


def main(ff_type, input_size):
    tf.reset_default_graph()
    if ff_type=='magnitude':
        images1, labels1 = regressor_placeholders(input_size=input_size*input_size, output_size=1)
        logits1 = birnn(images1, dim_out=1, partition=ff_type)
    else:
        images1, labels1 = regressor_placeholders(input_size=input_size*input_size*2, output_size=2)
        logits1 = birnn(images1, dim_out=2, partition=ff_type)
    data1 = RNNDataSet(input_size=input_size, training=False, partition=ff_type, bidirection=True)
    true_Y1 = np.array(data1.get_test_label())
    true_Y1_mean = np.mean(true_Y1, axis=0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver, save_path = utils.restore(sess, get('birnn.'+ff_type+'_checkpoint'))
        pred1 = pred_rnn(sess, saver, save_path, images1, logits1, data1, ff_type)
    print("Coefficient of determination R^2 for BiRnn is:", \
            np.sum((pred1-true_Y1_mean)**2)/np.sum((true_Y1-true_Y1_mean)**2))
    print("Mean norm percentage error for BiRnn is:", \
            np.sum(np.linalg.norm(pred1-true_Y1, axis=1))/np.sum(np.linalg.norm(true_Y1, axis=1)))
    print("Mean square percentage error for BiRnn is:", \
            np.sum((pred1-true_Y1)**2)/np.sum((true_Y1)**2))

    tf.reset_default_graph()
    if ff_type=='magnitude':
        images2, labels2 = regressor_placeholders(input_size=input_size*input_size, output_size=1)
        logits2 = rnn(images2, lstm_size=256, dim_out=1, num_layers=3, partition=ff_type)
    else:
        images2, labels2 = regressor_placeholders(input_size=input_size*input_size*2, output_size=2)
        logits2 = rnn(images2, lstm_size=256, dim_out=2, num_layers=2, partition=ff_type)
    data2 = RNNDataSet(input_size=input_size, training=False, partition=ff_type, bidirection=False)
    true_Y2 = np.array(data2.get_test_label())
    true_Y2_mean = np.mean(true_Y2, axis=0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver, save_path = utils.restore(sess, get('rnn.'+ff_type+'_checkpoint'))
        pred2 = pred_rnn(sess, saver, save_path, images2, logits2, data2, ff_type)
    print("Coefficient of determination R^2 for Rnn is:", \
            np.sum((pred2-true_Y2_mean)**2)/np.sum((true_Y2-true_Y2_mean)**2))
    print("Mean norm percentage error for Rnn is:", \
            np.sum(np.linalg.norm(pred2-true_Y2, axis=1))/np.sum(np.linalg.norm(true_Y2, axis=1)))
    print("Mean square percentage error for Rnn is:", \
            np.sum((pred2-true_Y2)**2)/np.sum((true_Y2)**2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ff_type', required=True)
    parser.add_argument('--input_size', required=True)
    args = parser.parse_args()
    main(args.ff_type, int(args.input_size))
