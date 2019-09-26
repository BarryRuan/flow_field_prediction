"""
Train RNN 
    Trains a recurrent neural network to predict vectors
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_rnn.py
"""
import argparse
import tensorflow as tf
import utils
from data.RNNDataSet import RNNDataSet 
from model.build_rnn import rnn
from model.build_birnn import birnn
from model.build_birnn import multilayer_birnn
from train_common import *
from utils import get

def report_training_progress(
    sess, batch_index, images, labels, loss, acc, food):
    """
    Performs inference on the validation set and reports the loss
    to the terminal and the training plot.
    """
    if batch_index % 50 == 0:
        batch_images, batch_labels = food.get_batch(
            partition='valid', batch_size=get('rnn.batch_size'))
        valid_acc, valid_loss = sess.run(
            [acc, loss],
            feed_dict={images : batch_images, labels : batch_labels})
        utils.log_training(batch_index, valid_loss, valid_acc)
        utils.update_training_plot(batch_index, valid_acc, valid_loss)

def train_rnn(
    sess, saver, save_path, images, labels, loss, train_op, acc, food):
    """
    Trains a tensorflow model of a cnn to classify a labeled image dataset.
    Periodically saves model checkpoints and reports the network
    performance on a validation set.
    """
    utils.make_training_plot()
    errors = 0
    for batch_index in range(get('rnn.num_steps')):
        report_training_progress(
            sess, batch_index, images, labels, loss, acc, food)
        # Run one step of training
        batch_images, batch_labels = food.get_batch(
            partition='train', batch_size=get('rnn.batch_size'))
        err,_ = sess.run([loss, train_op], feed_dict={images: batch_images, labels: batch_labels})
        errors += err
        # Save model parameters periodically
        if batch_index % 50 == 0:
            saver.save(sess, save_path)
            print("        Training loss =", errors/50)
            errors = 0

def main(ff_type, bi, input_size):
    print('======building model...======')
    if ff_type == 'magnitude':
        images, labels = regressor_placeholders(input_size*input_size, 1)
        if bi == 'true':
            logits = birnn(images, dim_out=1, partition=ff_type)
        else:
            logits = rnn(images, lstm_size=256, dim_out=1, num_layers=3, partition=ff_type)
        loss = mean_squared_error(labels, logits)
    else:
        images, labels = regressor_placeholders(input_size*input_size*2, 2)
        if bi == 'true':
            logits = birnn(images, dim_out=2, partition=ff_type)
        else:
            logits = rnn(images, lstm_size=256, dim_out=2, num_layers=2, partition=ff_type)
        loss = mean_squared_error(labels, logits)
    acc = regressor_accuracy(labels, logits, partition=ff_type)
    train_op = unsupervised_optimizer(loss, lr=3e-4)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if bi == 'true':
            saver, save_path = utils.restore(sess, get('birnn.'+ff_type+'_checkpoint'))
            data = RNNDataSet(input_size=input_size, partition=ff_type, bidirection=True)
        else:
            saver, save_path = utils.restore(sess, get('rnn.'+ff_type+'_checkpoint'))
            data = RNNDataSet(input_size=input_size, partition=ff_type)
        train_rnn(
            sess, saver, save_path, images,
            labels, loss, train_op, acc, data)
        print('=======saving trained model...======\n')
        saver.save(sess, save_path)
        utils.hold_training_plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ff_type', required=True)
    parser.add_argument('--bi', required=True)
    parser.add_argument('--input_size', required=True)
    args = parser.parse_args()
    main(args.ff_type, args.bi, int(args.input_size))
