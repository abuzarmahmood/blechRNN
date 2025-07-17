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
import os
file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(file_path)
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

class smooth_MSELoss(nn.Module):
    """
    MSE loss with temporal smoothness constraint
    """

    def __init__(self, alpha=0.05):
        super(smooth_MSELoss, self).__init__()
        self.loss1 = nn.MSELoss()
        self.alpha = alpha

    def mean_diffrence(self, x):
        """
        Calculate the mean difference between adjacent elements

        Args:
            x: (seq_len, batch, output_size)
        """
        return torch.mean(torch.abs(x[1:] - x[:-1])) * self.alpha

    def forward(self, input, target):
        """
        Args:
            input: (seq_len,batch, output_size)
            target: (seq_len,batch, output_size)
        """
        loss = self.loss1(input, target) + self.mean_diffrence(input)
        return loss

def MSELoss():
    return nn.MSELoss()

def train_model(
        net, 
        inputs, 
        labels, 
        output_size,
        train_steps = 1000, 
        lr=0.01, 
        delta_loss = 0.01,
        device = None,
        criterion = MSELoss(), 
        test_inputs = None,
        test_labels = None,
        save_path='best_model.pt',  # New parameter for saving model
        patience=10  # New parameter for early stopping
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

    cross_val_bool = np.logical_and(
            test_inputs is not None, 
            test_labels is not None
            )

    loss_history = []
    cross_val_loss = {}
    best_loss = float('inf')
    patience_counter = 0
    running_loss = 0
    running_acc = 0
    start_time = time.time()
    best_cross_val_loss = float('inf')  # Initialize best loss
    print('Training network...')
    for i in range(train_steps):
        # labels = labels.reshape(-1, output_size)


        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        # # Reshape to (SeqLen x Batch, OutputSize)
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

            # Save best model before checking for early stopping
            cross_str = f'Cross Val Loss: {test_loss.item():0.4f}'
            if test_loss.item() < best_cross_val_loss:
                best_cross_val_loss = test_loss.item()
                torch.save(net.state_dict(), save_path)  # Save best model
            cross_str = ''
            cross_val_loss[i] = test_loss.item()

            # Early stopping check
            if patience is not None:
                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f'Early stopping at step {i+1}')
                    break


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


