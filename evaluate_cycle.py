import numpy as np
import random
from model.build_rnn import rnn 
from model.build_birnn import birnn 
import matplotlib.pyplot as plt
import argparse
import os
from utils import get
import utils
from train_common import *
import time


def visualize_inputs(ff,ff_type,bi):
    size=ff.shape[1]
    if ff_type == 'magnitude':
        x, y = np.meshgrid(np.arange(size),np.arange(size-1,-1,-1))
        i = 1
        for true_ff in ff:
            z = true_ff
            if bi:
                plt.subplot(2,5,i)
            else:
                plt.subplot(3,3,i)
            plt.contourf(x,y,z,10, alpha=.75, cmap='jet')
            plt.colorbar()
            i += 1
        plt.show()
    else:
        x, y = np.meshgrid(np.arange(size+2),np.arange(size+1,-1,-1))
        i = 1
        for true_ff in ff:
            u = true_ff[:,:,0]
            v = true_ff[:,:,1]
            u1 = np.zeros((size+2)*(size+2)).reshape(size+2,size+2)
            v1 = np.zeros((size+2)*(size+2)).reshape(size+2,size+2)
            u1[1:1+size,1:1+size] = u
            v1[1:1+size,1:1+size] = v
            if bi:
                plt.subplot(2,5,i)
            else:
                plt.subplot(3,3,i)
            if ff_type=='velocity':
                plt.quiver(x,y,u1,v1, scale=190)
            else:
                plt.quiver(x,y,u1,v1, scale=20)
            i += 1
        plt.show()
     

def predict_rnn(ff_type, ff, input_size, bi):
    hs = input_size//2

    pred = []
    testX = []
    for i in range(8,17):
        for j in range(8,17):
            if bi:
                if ff_type == 'magnitude':
                    testX.append(np.concatenate([ff[:5, i-hs:i+hs+1, j-hs:j+hs+1],
                        ff[6:, i-hs:i+hs+1, j-hs:j+hs+1]]).reshape(9,-1))
                else:
                    testX.append(np.concatenate([ff[:5, i-hs:i+hs+1, j-hs:j+hs+1, :],
                        ff[6:, i-hs:i+hs+1, j-hs:j+hs+1, :]]).reshape(9,-1))
            else:
                if ff_type == 'magnitude':
                    testX.append(ff[:9, i-hs:i+hs+1, j-hs:j+hs+1].reshape(9,-1))
                else:
                    testX.append(ff[:9, i-hs:i+hs+1, j-hs:j+hs+1, :].reshape(9,-1))
    testX = np.array(testX)

    tf.reset_default_graph()
    if ff_type == 'magnitude':
        images, labels = regressor_placeholders(input_size=input_size*input_size,
                output_size=1, num_step=9)
        if bi:
            logits = birnn(images, lstm_size=500, dim_out=1, partition=ff_type)
        else:
            logits = rnn(images, lstm_size=256, dim_out=1,num_layers=3, partition=ff_type)
    else:
        images, labels = regressor_placeholders(input_size=input_size*input_size*2,
                output_size=2, num_step=9)
        if bi:
            logits = birnn(images, lstm_size=500, dim_out=2, partition=ff_type)
        else:
            logits = rnn(images, lstm_size=256,  dim_out=2, num_layers=2, partition=ff_type)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if bi:
            saver, save_path = utils.restore(sess, get('birnn.'+ff_type+'_checkpoint'))
        else:
            saver, save_path = utils.restore(sess, get('rnn.'+ff_type+'_checkpoint'))
        t0 = time.time()
        pred = sess.run(logits, feed_dict={images: testX.swapaxes(0,1)})
        print("It takes %s second to predict the target flow field." % (str(round(time.time()-t0, 5))))
    return pred

def visualize_seq(ff_type, ff, bi):
    if ff_type == 'magnitude':
        if bi:
            visualize_inputs(ff[:, 8:17, 8:17], ff_type, bi)
        else:
            visualize_inputs(ff[:9, 8:17, 8:17], ff_type, bi)
    else:
        if bi:
            visualize_inputs(ff[:, 8:17, 8:17, :], ff_type, bi)
        else:
            visualize_inputs(ff[:9, 8:17, 8:17, :], ff_type, bi)

def read_inputs(ff_type, a, start_ff, bi):
    ff = []
    if ff_type == 'magnitude':
        for choice in range(a+start_ff*100,1000+start_ff*100,100):
            ff.append(utils.read_mag_field(os.path.join('data/'+ff_type, str(choice))))
        ff=np.array(ff)
        if bi:
            true_ff = ff[5, 8:17, 8:17]
        else:
            true_ff = ff[9, 8:17, 8:17]
    else:
        for choice in range(a+start_ff*100,1000+start_ff*100,100):
            ff.append(utils.read_flow_field(os.path.join('data/'+ff_type, str(choice))))
        ff=np.array(ff)
        if bi:
            true_ff = ff[5, 8:17, 8:17, :]
        else:
            true_ff = ff[9, 8:17, 8:17, :]
    return ff, true_ff


def main(ff_type, input_size, bi):
    hs = input_size//2
    isq = input_size*input_size
    isq = 81 
    ff = []
    # num_cycle = random.randint(1,99)
    num_cycle = 80
    print("Number of cycle is:", num_cycle)
    start_ff = random.randint(10,90)
    los = []
    adj_dev = []
    for start_ff in range(0,91,10):
        #42, 81, 90, 78
        print("Start flow field is No.", start_ff*100+num_cycle)
        if bi == 'true':
            print("Target flow field is No.", start_ff*100+500+num_cycle)
            ff, true_ff = read_inputs(ff_type, num_cycle, start_ff, True)
            pred = predict_rnn(ff_type, ff, input_size, bi=True)
            # visualize_seq(ff_type, ff, bi=True)
        else:
            print("Target flow field is No.", start_ff*100+900+num_cycle)
            ff, true_ff = read_inputs(ff_type, num_cycle, start_ff, False)
            pred = predict_rnn(ff_type, ff, input_size, bi=False)
            # visualize_seq(ff_type, ff, bi=False)
        mnd = np.nanmean(np.linalg.norm(pred-true_ff.reshape(isq,2), axis=1))
        print("Mean normed distance is:", mnd)
        mag = np.linalg.norm(true_ff.reshape(isq,2), axis=1).reshape(1,-1)
        print("average magnitude for vectors in the target flow field is:", np.nanmean(mag))
        distance = np.linalg.norm(pred-true_ff.reshape(isq,2), axis=1).reshape(1,-1)
        per_error = np.sum(distance)/np.sum(mag)
        print("Percentage error:", per_error)
        los.append(per_error)
        """"
        if ff_type == 'velocity':
            pred = pred.reshape(9,9,2)
            utils.visualize_result(true_ff, pred, ff_type, pred)
        elif ff_type == 'magnitude':
            pred = pred.reshape(9,9)
            utils.visualize_result(true_ff, pred, ff_type)
        else:
            pred = pred.reshape(9,9,2)
            utils.visualize_result(true_ff, pred, ff_type, pred)
        """
    plt.plot(los)
    plt.show()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ff_type', required=True)
    parser.add_argument('--input_size', required=True)
    parser.add_argument('--bi', required=True)
    args = parser.parse_args()
    
    main(args.ff_type, int(args.input_size), args.bi)
