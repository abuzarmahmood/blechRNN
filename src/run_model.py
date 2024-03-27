"""
Code to run the model on the data
and generate outputs
"""

############################################################
# Load data
import sys
import os
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz

plot_dir = '/media/bigdata/firing_space_plot/modelling/pytorch_rnn/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

artifacts_dir = '/media/bigdata/firing_space_plot/modelling/pytorch_rnn/artifacts'
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

data_dir = '/media/storage/gc_only/AM34/AM34_4Tastes_201217_114556' 
dat = ephys_data(data_dir)
dat.firing_rate_params = dat.default_firing_params
dat.get_spikes()
dat.get_firing_rates()

spike_array = np.stack(dat.spikes)
# Drop first n trials
spike_n_trials = 0
spike_array = spike_array[:,spike_n_trials:]

cat_spikes = np.concatenate(spike_array)

# Bin spikes
bin_size = 25
binned_spikes = np.reshape(cat_spikes, 
                           (*cat_spikes.shape[:2], -1, bin_size)).sum(-1)

# vz.firing_overview(binned_spikes.swapaxes(0,1))
# plt.show()

# Reshape to (seq_len, batch, input_size)
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

# Scale the PCA outputs
pca_scaler = StandardScaler()
inputs_pca = pca_scaler.fit_transform(inputs_pca)

inputs_trial_pca = inputs_pca.reshape(inputs.shape[0], -1, n_components)
inputs = inputs_trial_pca.copy()

##############################

# Add taste number as external input
taste_number = np.repeat(np.arange(4), inputs.shape[1]//4)
taste_number = np.broadcast_to(taste_number, (inputs.shape[0], inputs.shape[1]))

# Add stim time as external input
stim_time = np.zeros_like(taste_number)
stim_time[:, 2000//bin_size] = 1

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

input_size = inputs.shape[-1] + 1 # Add 2 for taste, and stim-t
output_size = inputs.shape[-1] 

# Instead of predicting activity in the SAME time-bin,
# predict activity in the NEXT time-bin
# Hoping that this will make the model more robust to
# small fluctuations in activity
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

hidden_size_vec = [6, 8]
loss_funcs_vec = ['mse']
repeats = 3

param_product = [[[x,y]]*repeats for x in hidden_size_vec for y in loss_funcs_vec]
param_product = [item for sublist in param_product for item in sublist]

# n_restarts = len(hidden_size_vec)
model_list = []
loss_list = []
cross_val_loss_list = []
# for i in trange(n_restarts):
for i, this_params in enumerate(tqdm(param_product)):
    net = autoencoderRNN( 
            input_size=input_size,
            hidden_size= this_params[0], 
            output_size=output_size,
            dropout = 0.2,
            # rnn_layers = 3,
            )
    net.to(device)
    net, loss, cross_val_loss = train_model(
            net, 
            train_inputs, 
            train_labels, 
            lr = 0.001, 
            train_steps = 15000,
            loss = this_params[1], 
            test_inputs = test_inputs,
            test_labels = test_labels,
            )
    model_list.append(net)
    loss_list.append(loss)
    cross_val_loss_list.append(cross_val_loss)

    model_name = f'hidden_{this_params[0]}_loss_{this_params[1]}_{i}'
    torch.save(net, os.path.join(artifacts_dir, f'{model_name}.pt'))

    fig, ax = plt.subplots()
    # ax.plot(np.stack(loss_list).T, alpha = 0.5)
    # ax.set_title(f'Losses for {i} restarts')
    for ind, loss in enumerate(loss_list):
        ax.plot(loss, label = f'params: {param_product[ind]}') 
    ax.legend(
            bbox_to_anchor=(1.05, 1), 
            loc='upper left', borderaxespad=0.)
    ax.set_title(f'Losses for {i+1}/{len(param_product)} restarts')
    fig.savefig(os.path.join(plot_dir,'run_loss.png'),
                bbox_inches = 'tight')
    plt.close(fig)

    # Plot another figure of cross_val_loss
    fig, ax = plt.subplots()
    for ind, loss in enumerate(cross_val_loss_list):
        ax.plot(loss.keys(), loss.values(),
                label = f'params: {param_product[ind]}')
    ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left', borderaxespad=0.)
    ax.set_title(f'Cross Val Losses for {i+1}/{len(param_product)} restarts')
    fig.savefig(os.path.join(plot_dir,'cross_val_loss.png'),
                bbox_inches = 'tight')
    plt.close(fig)

