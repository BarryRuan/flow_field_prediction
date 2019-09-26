"""
Utility functions
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as m

def restore(sess, checkpoint_path):
    """
    If a checkpoint exists, restores the tensorflow model from the checkpoint.
    Returns the tensorflow Saver and the checkpoint filename.
    """
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    if checkpoint:
        path = checkpoint.model_checkpoint_path
        print('Restoring model parameters from {}'.format(path))
        saver.restore(sess, path)
    else:
        print('No saved model parameters found')
    # Return checkpoint path for call to saver.save()
    save_path = os.path.join(
        checkpoint_path, os.path.basename(os.path.dirname(checkpoint_path)))
    return saver, save_path


def get(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(get, 'config'):
        with open('config.json') as f:
            get.config = eval(f.read())
    node = get.config
    for part in attr.split('.'):
        node = node[part]
    return node

def log_training(batch_index, valid_loss, valid_acc=None):
    """
    Logs the validation accuracy and loss to the terminal
    """
    print('Batch {}'.format(batch_index))
    if valid_acc != None:
        print('\tValidation loss: {}'.format(valid_loss))
        print('\tAccuracy: {}'.format(valid_acc))
    else:
        print('\tMean squared error loss: {}'.format(valid_loss))

def make_training_plot():
    """
    Runs the setup for an interactive matplotlib graph that logs the loss and
    accuracy
    """
    plt.ion()
    plt.title('Supervised Network Training')
    plt.subplot(1, 2, 1)
    plt.xlabel('Batch Index')
    plt.ylabel('Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.xlabel('Batch Index')
    plt.ylabel('Validation Loss')

def make_ae_training_plot():
    """
    Runs the setup for an interactive matplotlib graph that logs the loss
    """
    plt.ion()
    plt.title('Autoencoder Training')
    plt.xlabel('Batch Index')
    plt.ylabel('Validation MSE')

def update_training_plot(batch_index, valid_acc, valid_loss):
    """
    Updates the training plot with a new data point for loss and accuracy
    """
    plt.subplot(1, 2, 1)
    plt.scatter(batch_index, valid_acc, c='b')
    plt.subplot(1, 2, 2)
    plt.scatter(batch_index, valid_loss, c='r')
    plt.pause(0.00001)

def update_ae_training_plot(batch_index, valid_loss):
    """
    Updates the training plot with a new data point for loss
    """
    plt.scatter(batch_index, valid_loss, c='r')
    plt.pause(0.00001)

def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()


def denormalize_image(image):
    """ Rescale the image's color space from (min, max) to (0, 1) """
    ptp = np.max(image, axis=(0,1)) - np.min(image, axis=(0,1))
    return (image - np.min(image, axis=(0,1))) / ptp


def read_flow_field(filename):
    """
    Read the matrix representation of the velocity flow field given by "filename"
    """
    f = open(filename)
    lines = f.readlines()
    f.close()
    v = []
    for line in lines:
        v.append([[float(b) for b in a.split(',')] for a in line.split(' ')])
    return v

def read_mag_field(filename):
    """
    Read the matrix representation of the magnitude flow field given by "filename"
    """
    f = open(filename)
    lines = f.readlines()
    f.close()
    v = []
    for line in lines:
        v.append([float(a) for a in line.split(' ')])
    return v


def visualize_result(true_ff, pred, ff_type, true_v=None):
    size=true_ff.shape[0]
    if ff_type == 'magnitude':
        cdict = {
  'red'  :  ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
  'green':  ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
  'blue' :  ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))}
        cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
        x, y = np.meshgrid(np.arange(size),np.arange(size-1,-1,-1))
        z1 = true_ff
        z2 = pred
        z3 = true_v
        plt.subplot(1,2,1)
        plt.title("True flow flield")
        plt.contourf(x,y,z1,20, alpha=.75, cmap='jet', vmin=0, vmax=16)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.title("Predicted flow flield")
        plt.contourf(x,y,z2,20, alpha=.75, cmap='jet', vmin=0, vmax=16)
        plt.colorbar()
        """
        plt.subplot(1,3,3)
        plt.contourf(x,y,z3,20, alpha=.75, cmap='jet')
        """
        plt.show()
    else:
        x, y = np.meshgrid(np.arange(size+2),np.arange(size+1,-1,-1))
        u1=np.zeros((size+2)*(size+2)).reshape(size+2,size+2)
        v1=np.zeros((size+2)*(size+2)).reshape(size+2,size+2)
        u2=np.zeros((size+2)*(size+2)).reshape(size+2,size+2)
        v2=np.zeros((size+2)*(size+2)).reshape(size+2,size+2)
        u3=np.zeros((size+2)*(size+2)).reshape(size+2,size+2)
        v3=np.zeros((size+2)*(size+2)).reshape(size+2,size+2)
        u1[1:1+size,1:1+size] = true_ff[:,:,0]
        v1[1:1+size,1:1+size] = true_ff[:,:,1]
        if ff_type=='velocity':
            if not type(true_v) == np.ndarray:
                plt.subplot(1,2,1)
            else:
                plt.subplot(1,3,1)
            plt.quiver(x,y,u1,v1, scale=110, label='True flow field')
        else:
            plt.subplot(2,2,1)
            plt.quiver(x,y,u1,v1, scale=20, label='True flow field')
        plt.title("True flow field")
        u2[1:1+size,1:1+size] = pred[:,:,0]
        v2[1:1+size,1:1+size] = pred[:,:,1]
        if ff_type=='velocity':
            if not type(true_v) == np.ndarray:
                plt.subplot(1,2,2)
            else:
                plt.subplot(1,3,2)
            plt.quiver(x,y,u2,v2, scale=110, label='Predicted flow field')
        else:
            plt.subplot(2,2,2)
            plt.quiver(x,y,u2,v2, scale=20, label='Predicted flow field')
        plt.title("Predicted flow field")
        if ff_type=='direction':
            u3[1:1+size,1:1+size] = true_v[:,:,0]
            v3[1:1+size,1:1+size] = true_v[:,:,1]
            plt.subplot(2,2,3)
            plt.quiver(x,y,u3,v3, scale=160, label='True velocity map')
            plt.title("Actual velocity map")
        true_v=true_ff-pred
        if type(true_v) == np.ndarray:
            u3[1:1+size,1:1+size] = true_v[:,:,0]
            v3[1:1+size,1:1+size] = true_v[:,:,1]
            if ff_type=='velocity':
                plt.subplot(1,3,3)
                plt.quiver(x,y,u3,v3, scale=110, label='Predicted by normal rnn')
            else:
                plt.subplot(2,2,4)
                plt.quiver(x,y,u3,v3, scale=20, label='True velocity map')
        plt.title("Residual flow field")
        plt.show()



