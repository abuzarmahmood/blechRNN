"""
Class to handle model training and prediction
""" 

############################################################
# Imports
############################################################

import time
import numpy as np
import pylab as plt
from tqdm import tqdm, trange
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import explained_variance_score, r2_score
import sys
src_dir = '/media/bigdata/firing_space_plot/modelling/pytorch_rnn/src'
sys.path.append(src_dir)
from model import autoencoderRNN

############################################################
# Define Model 
############################################################
# Define networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.nn import functional as F
import math
from scipy.stats import poisson, zscore


def train_model(
        net, 
        inputs, 
        labels, 
        output_size,
        train_steps = 1000, 
        lr=0.01, 
        delta_loss = 0.01,
        device = None,
        loss = 'poisson',
        test_inputs = None,
        test_labels = None,
        ):
    """Simple helper function to train the model.

    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair
        inputs: shape (seq_len, batch, input_size)
        labels: shape (seq_len * batch, output_size)

    Returns:
        net: network object after training
    """
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    cross_val_bool = np.logical_and(
            test_inputs is not None, 
            test_labels is not None
            )

    loss_history = []
    cross_val_loss = {}
    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    print('Training network...')
    for i in range(train_steps):
        labels = labels.reshape(-1, output_size)


        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        output = output.reshape(-1, output_size)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()    # Does the update

        # Only compute cross_val_loss every 100 steps
        # because it's expensive
        if cross_val_bool and (i % 100 == 99): 
            test_out, _ = net(test_inputs)
            test_out = test_out.reshape(-1, output_size)
            test_labels = test_labels.reshape(-1, output_size)
            test_loss = criterion(test_out, test_labels)
            cross_val_loss[i] = test_loss.item()
            cross_str = f'Cross Val Loss: {test_loss.item():0.4f}'
        else:
            cross_str = ''

        # Compute the running loss every 100 steps
        current_loss = loss.item()
        loss_history.append(current_loss)
        running_loss += current_loss 
        if i % 100 == 99:
            running_loss /= 100
            print('Step {}, Loss {:0.4f}, {}, Time {:0.1f}s'.format(
                i+1, running_loss, cross_str, time.time() - start_time))
            running_loss = 0
    return net, loss_history, cross_val_loss


