import numpy as np
import pandas as pd
from scipy.misc import imread, imresize
import os
import sys
from utils import get
import random
import matplotlib.pyplot as plt

class RNNDataSet:

    def __init__(self, training=True, input_size=9, seq_len=9, 
            partition='velocity', use_auto=False, bidirection=False):
        """
        Reads in the necessary data from disk and prepares data for training.
        """
        np.random.seed(0)
        self.input_size=input_size
        self.seq_len=seq_len
        self.partition=partition
        self.use_auto=use_auto
        if bidirection:
            self.target_ff=6
        else:
            self.target_ff=0

        # Load in all the data we need from disk

        if training:
            self.trainX, self.trainY = self._load_data('train')
            self.train_count = 0
            self.validX, self.validY = self._load_data('valid')
            self.valid_count = 0
        else:
            self.testX, self.testY = self._load_data('test')
            self.test_count = 0

    def get_test_label(self):
        return self.testY

    def get_batch(self, partition, batch_size=100):
        """
        Returns a batch of batch_size examples. If partition is not test,
        also returns the corresponding labels.
        """
        if partition == 'train':
            batchX, batchY, self.trainX, self.trainY, self.train_count = \
                self._batch_helper(
                    self.trainX, self.trainY, self.train_count, batch_size)
            return batchX, batchY
        elif partition == 'valid':
            batchX, batchY, self.validX, self.validY, self.valid_count = \
                self._batch_helper(
                    self.validX, self.validY, self.valid_count, batch_size)
            return batchX, batchY
        elif partition == 'test':
            batchX, self.testX, self.test_count = \
                self._batch_helper(
                    self.testX, None, self.test_count, batch_size)
            return batchX
        else:
            raise ValueError('Partition {} does not exist'.format(partition))

    def finished_test_epoch(self):
        """
        Returns true if we have finished an iteration through the test set.
        Also resets the state of the test counter.
        """
        result = self.test_count >= len(self.testX)
        if result:
            self.test_count = 0
        return result

    def _batch_helper(self, X, y, count, batch_size):
        """
        Handles batching behaviors for all data partitions, including data
        slicing, incrementing the count, and shuffling at the end of an epoch.
        Returns the batch as well as the new count and the dataset to maintain
        the internal state representation of each partition.
        """
        if count + batch_size > len(X):
            if type(y) == np.ndarray:
                count = 0
                rand_idx = np.random.permutation(len(X))
                X = X[rand_idx]
                y = y[rand_idx]
        batchX = X[count:count+batch_size]
        if type(y) == np.ndarray:
            batchY = y[count:count+batch_size]
        count += batch_size
        if type(y) == np.ndarray:
            return batchX.swapaxes(0,1), batchY, X, y, count
        else:
            return batchX.swapaxes(0,1), X, count


    def _load_data(self, partition='train'):
        """
        Loads a single data partition from file.
        """
        print("======loading %s...======" % partition)
        X, Y = self._get_ff_and_labels(partition)
        return X, Y


    def _get_ff_and_labels(self, partition='train'):
        """
        Fetches the data based on image filenames specified in df.
        If training is true, also loads the labels.
        """
        dir_name = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
        ss = self.input_size * self.input_size
        dir_name1 = os.path.join(dir_name, get('time_'+self.partition+'_path'))
        dir_name2 = os.path.join(dir_name, get('compressed_time_'+self.partition+'_path'))
        if partition == 'test':
            X, y = [], []
            if not self.use_auto:
                dir_name = dir_name1
            else:
                dir_name = dir_name2
            mag_list = os.listdir(dir_name)
            for mag in mag_list:
                if mag[:4] == 'test':
                    f = open(os.path.join(dir_name,mag), 'r')
                    lines = f.readlines()
                    f.close()
                    if self.use_auto:
                        f1 = open(os.path.join(dir_name1,mag))
                        f1_lines = f1.readlines()
                    ff = []
                    count = 0
                    for index, line in enumerate(lines):
                        count += 1
                        line_elements = [float(a) for a in line.split(',')]
                        if not count % (self.seq_len+1) == self.target_ff:
                            ff.append(line_elements)
                        else:
                            if not self.use_auto:
                                if not self.partition == 'magnitude':
                                    y.append([line_elements[ss-1], line_elements[ss]])
                                else:
                                    y.append([line_elements[ss//2]])
                            else:
                                ll = [float(a) for a in f1_lines[index].split(',')]
                                if not self.partition == 'magnitude':
                                    y.append([ll[ss-1], ll[ss]])
                                else:
                                    y.append([ll[ss//2]])
                    X = X + np.array(ff).reshape(10000//(self.seq_len+1), self.seq_len, -1).tolist()
            return np.array(X), np.array(y)
        else:
            X, y = [], []
            if not self.use_auto:
                dir_name = dir_name1
            else:
                dir_name = dir_name2
            mag_list = os.listdir(dir_name)
            for mag in mag_list:
                if mag[:5] == partition:
                    print("Reading", mag)
                    f = open(os.path.join(dir_name,mag), 'r')
                    lines = f.readlines()
                    f.close()
                    if self.use_auto:
                        f1 = open(os.path.join(dir_name1,mag))
                        f1_lines = f1.readlines()
                    ff = []
                    count = 0
                    for index,line in enumerate(lines):
                        count += 1
                        line_elements = [float(a) for a in line.split(',')]
                        if not count % (self.seq_len+1) == self.target_ff:
                            ff.append(line_elements)
                        else:
                            if not self.use_auto:
                                if not self.partition == 'magnitude':
                                    y.append([line_elements[ss-1], line_elements[ss]])
                                else:
                                    y.append([line_elements[ss//2]])
                            else:
                                ll = [float(a) for a in f1_lines[index].split(',')]
                                if not self.partition == 'magnitude':
                                    y.append([ll[ss-1], ll[ss]])
                                else:
                                    y.append([ll[ss//2]])
                    X = X + np.array(ff).reshape(10000//(self.seq_len+1), self.seq_len, -1).tolist()
            z = list(zip(X, y))
            random.shuffle(z)
            X[:], y[:] = zip(*z)
            return np.array(X), np.array(y)
