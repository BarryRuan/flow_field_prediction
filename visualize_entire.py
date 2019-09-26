"""
Visulize the entire flow field.
Usage: python3 visualize_entire.py --ff_type='velocity' --filename=9000
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import utils

def visualize_2d(filename, ff_type='velocity'):
    """
    Plot a quiver map for its filename
    """
    path = os.path.join(os.path.join('data/', ff_type), filename)
    ff = open(path)
    lines = ff.readlines()
    ff.close()
    v = []
    for line in lines:
        v.append([[float(b) for b in a.split(',')] for a in line.split(' ')])
    v = np.array(v)
    x, y = np.meshgrid(np.arange(25),np.arange(24,-1,-1))
    u1 = v[:,:,0]
    v1 = v[:,:,1]
    if ff_type == 'velocity':
        plt.quiver(x,y,u1,v1, scale=500)
        plt.title("Velocity Map")
    else:
        plt.quiver(x,y,u1,v1, scale=50)
        plt.title("Direction Map")
    plt.show()

def visualize_1d(a):
    """
    Plot a contour plot for filename a and a quiver map for its velocity map
    """
    true_v=utils.read_flow_field(os.path.join('data/velocity', str(a)))
    ff=utils.read_mag_field(os.path.join('data/magnitude', str(a)))
    ff=np.array(ff)
    true_v=np.array(true_v)
    size=ff.shape[0]
    x,y = np.meshgrid(np.array([i for i in range(25)]), np.array([i for i in range(24,-1,-1)]))
    z = ff 
    plt.subplot(1,2,1)
    plt.contourf(x,y,z,10, alpha=.75, cmap='jet')
    plt.colorbar()
    plt.title("Magnitude Map")

    x, y = np.meshgrid(np.arange(25),np.arange(24,-1,-1))
    u = true_v[:,:,0]
    v = true_v[:,:,1]
    plt.subplot(1,2,2)
    plt.quiver(x,y,u,v, scale=300)
    plt.title("Velocity Map")
    plt.show()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True)
    parser.add_argument('--ff_type', required=True)
    args = parser.parse_args()
    if args.ff_type=='magnitude':
        visualize_1d(args.filename)
    else:
        visualize_2d(args.filename, args.ff_type)
