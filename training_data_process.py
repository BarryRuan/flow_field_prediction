"""
Preprocess the training data
Write new training files so that each file specifies a 
certain location in the velocity field with all the corresponding
windows for all the training flow fields. Each line in these file
represent a flow field.
"""
import numpy as np
import argparse
import math
import os
from utils import get
from utils import *
import utils

# Location we focus on
valid_location = [[8,i] for i in range(8,17)] + \
        [[9,i] for i in range(8,17)] + \
        [[10,i] for i in range(8,17)] + \
        [[11,i] for i in range(8,17)] + \
        [[12,i] for i in range(8,17)] + \
        [[13,i] for i in range(8,17)] + \
        [[14,i] for i in range(8,17)] + \
        [[15,i] for i in range(8,17)] + \
        [[16,i] for i in range(8,17)]


def process_training_data(input_size):
    # *_ffs stores all the data we have
    m_ff = []
    v_ff = []
    d_ff = []

    for i in range(1,20001):
        print('read', str(i))
        v_ff.append(utils.read_flow_field('data/velocity/%d'%(i)))
        d_ff.append(utils.read_flow_field('data/direction/%d'%(i)))
        m_ff.append(utils.read_mag_field('data/magnitude/%d'%(i)))
    # list to np.array
    m_ff = np.array(m_ff)
    v_ff = np.array(v_ff)
    d_ff = np.array(d_ff)

    # half size of input size
    hs = (input_size-1)//2

    # dictionary whoes key is string repr of the location and value
    # is input_size*input_size windows centered at the location of
    # all the flow fields we have.
    m_ff_window = {} 
    v_ff_window = {} 
    d_ff_window = {} 
    for loc in valid_location:
        loc_string = '_'.join([str(loc[0]), str(loc[1])])
        m_ff_window[loc_string] = []
        v_ff_window[loc_string] = []
        d_ff_window[loc_string] = []

        # We have totally 200 cycles, 100 for high swirl and 100 
        # for low swirl, respectively. The first 10000 flow fields
        # are for high swirl flow fields , so if the cycle is from 
        # high swirl, for cycle i, filenames 100*n+i, n=0,...,99 are
        # flow fields derived from the cycle. After the following
        # operations, each 100 sequential flow fields are from the 
        # same cycle.
        for choice in range(100):
            for ff_num in range(choice,10000,100):
                m_ff_window[loc_string].append(m_ff[ff_num,loc[0]-hs:loc[0]+hs+1,loc[1]-hs:loc[1]+hs+1])
                v_ff_window[loc_string].append(v_ff[ff_num,loc[0]-hs:loc[0]+hs+1,loc[1]-hs:loc[1]+hs+1, :])
                d_ff_window[loc_string].append(d_ff[ff_num,loc[0]-hs:loc[0]+hs+1,loc[1]-hs:loc[1]+hs+1, :])
    return m_ff_window, v_ff_window, d_ff_window


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', required=True)
    args = parser.parse_args()
    input_size = int(args.input_size)
    m_ff_window, v_ff_window, d_ff_window =  process_training_data(input_size)
    for key in m_ff_window.keys():
        m_lines = np.array(m_ff_window[key]).reshape(-1, input_size*input_size)
        v_lines = np.array(v_ff_window[key]).reshape(-1, input_size*input_size*2)
        d_lines = np.array(d_ff_window[key]).reshape(-1, input_size*input_size*2)

        ran = np.random.randint(100)
        if ran < 80:
            key = 'train_'+key
        elif ran <= 90:
            key = 'valid_'+key
        else:
            key = 'test_'+key

        m_path = os.path.join(get('time_magnitude_path'), key+'_'+args.input_size)
        v_path = os.path.join(get('time_velocity_path'), key+'_'+args.input_size)
        d_path = os.path.join(get('time_direction_path'), key+'_'+args.input_size)
        # Write into corresponding files.
        print("writing", m_path)
        f = open(m_path, 'w')
        for line in m_lines:
            f.writelines(','.join([str(round(a,4)) for a in line])+'\n')
        f.close()
        print("writing", d_path)
        f = open(d_path, 'w')
        for line in d_lines:
            f.writelines(','.join([str(round(a,4)) for a in line])+'\n')
        f.close()
        print("writing", v_path)
        f = open(v_path, 'w')
        for line in v_lines:
            f.writelines(','.join([str(round(a,4)) for a in line])+'\n')
        f.close()

        




