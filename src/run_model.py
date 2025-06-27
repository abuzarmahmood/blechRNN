"""
Code to run the model on the data
and generate outputs
"""

############################################################
# Load data
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt

sys.path.append('/media/bigdata/firing_space_plot/modelling/blechRNN/src')
from model import autoencoderRNN
from train import train_model

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz

base_dir =  '/media/bigdata/firing_space_plot/modelling/pytorch_rnn/'
plot_dir = os.path.join(base_dir, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

artifacts_dir = os.path.join(base_dir, 'artifacts')
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191030_114043_copy'
dat = ephys_data(data_dir)
dat.firing_rate_params = dat.default_firing_params
dat.get_spikes()
dat.get_firing_rates()

spike_array = np.stack(dat.spikes)

cat_spikes = np.concatenate(spike_array)

# Bin spikes
bin_size = 25
# (tastes x trials, neurons, time)
# for example : (120, 35, 280)
binned_spikes = np.reshape(cat_spikes, 
                           (*cat_spikes.shape[:2], -1, bin_size)).sum(-1)

# Reshape to (seq_len, batch, input_size)
# seq_len = time
# batch = trials
# input_size = neurons
inputs = binned_spikes.copy()
inputs = np.moveaxis(inputs, -1, 0)

##############################
# Perform PCA on data
# If PCA is performed on raw data, higher firing neurons will dominate
# the latent space
# Therefore, perform PCA on zscored data

inputs_long = inputs.reshape(-1, inputs.shape[-1])

# Perform standard scaling
scaler = StandardScaler()
# scaler = MinMaxScaler()
inputs_long = scaler.fit_transform(inputs_long)

# Perform PCA and get 95% explained variance
pca_obj = PCA(n_components=0.95)
inputs_pca = pca_obj.fit_transform(inputs_long)
n_components = inputs_pca.shape[-1]

# Scale the PCA outputs
pca_scaler = StandardScaler()
inputs_pca = pca_scaler.fit_transform(inputs_pca)

inputs_trial_pca = inputs_pca.reshape(inputs.shape[0], -1, n_components)

# shape: (time, trials, pca_components)
inputs = inputs_trial_pca.copy()

##############################

# Add stim time as external input
# Shape: (time, trials, 1)
stim_time = np.zeros((inputs.shape[0], inputs.shape[1]))
stim_time[:, 2000//bin_size] = 1

# Shape: (time, trials, pca_components + 1)
inputs_plus_context = np.concatenate(
        [
            inputs, 
            stim_time[:,:,None]
            ], 
        axis = -1)

############################################################
# Train model
############################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = inputs_plus_context.shape[-1] 
output_size = inputs_plus_context.shape[-1] -1 # We don't want to predict the stim time

# Instead of predicting activity in the SAME time-bin,
# predict activity in the NEXT time-bin
# Forcing the model to learn temporal dependencies
inputs_plus_context = inputs_plus_context[:-1]
inputs = inputs[1:]

# (seq_len * batch, output_size)
labels = torch.from_numpy(inputs).type(torch.float32)
# (seq_len, batch, input_size)
inputs = torch.from_numpy(inputs_plus_context).type(torch.float)

# Split into train and test
train_test_split = 0.75
train_inds = np.random.choice(
        np.arange(inputs.shape[1]), 
        int(train_test_split * inputs.shape[1]), 
        replace = False)
test_inds = np.setdiff1d(np.arange(inputs.shape[1]), train_inds)

train_inputs = inputs[:,train_inds]
train_labels = labels[:,train_inds]
test_inputs = inputs[:,test_inds]
test_labels = labels[:,test_inds]


train_inputs = train_inputs.to(device)
train_labels = train_labels.to(device)
test_inputs = test_inputs.to(device)
test_labels = test_labels.to(device)

##############################
# Train 
##############################
# Hidden size of 8 was tested to be optimal across multiple datasets
hidden_size = 8
# mse loss performs better than poisson loss
loss_name = 'mse'

net = autoencoderRNN( 
        input_size=input_size,
        hidden_size= hidden_size, 
        output_size=output_size,
        rnn_layers = 2,
        dropout = 0.2,
        patience=10  # Set patience for early stopping
        )
net.to(device)
net, loss, cross_val_loss = train_model(
        net, 
        train_inputs, 
        train_labels, 
        output_size = output_size,
        lr = 0.001, 
        train_steps = 15000,
        loss = loss_name, 
        test_inputs = test_inputs,
        test_labels = test_labels,
        )

model_name = f'hidden_{hidden_size}_loss_{loss_name}'
torch.save(net, os.path.join(artifacts_dir, f'{model_name}.pt'))

# Make plots
fig, ax = plt.subplots()
ax.plot(loss, label = 'Train Loss') 
ax.plot(cross_val_loss.keys(), cross_val_loss.values(), label = 'Test Loss')
ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', borderaxespad=0.)
ax.set_title(f'Losses') 
fig.savefig(os.path.join(plot_dir,'run_loss.png'),
            bbox_inches = 'tight')
plt.close(fig)

