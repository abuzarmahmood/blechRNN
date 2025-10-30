"""
# # If on Google Colab, uncomment to install neurogym to use cognitive tasks
# ! git clone https://github.com/neurogym/neurogym.git
# %cd neurogym/
# ! pip install -e .

https://cbmm.mit.edu/video/tutorial-recurrent-neural-networks-cognitive-neuroscience

Recurrent neural network for firing rate estimation

Inputs:
    - spike trains (with binning) 
    - external input
Outputs:
    - firing rates

Loss:
    - Poisson log-likelihood

Initialization:
    - random weights
    - random biases
    - random initial conditions

Start prior to stim so initial conditions don't matter as much
"""

# Import common packages
import numpy as np
import matplotlib.pyplot as plt

# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math


class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms.
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Run network for one time step.

        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)

        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        steps = range(input.size(0)) # seq_len
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden


class CTRNN_plus_output(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size

    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)

        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output

class autoencoderRNN(nn.Module):
    """
    Input and output transformations are encoder and decoder architectures
    RNN will learn dynamics of latent space

    Output has to be rectified
    Can add dropout to RNN and autoencoder layers
    """
    def __init__(
            self, 
            input_size, 
            hidden_size,  
            output_size, 
            rnn_layers = 1,
            dropout = 0.2,
            bidirectional = False,
            strictly_positive = False,
            ):
        """
        3 sigmoid layers for input and output each, to project between:
            encoder : input -> latent
            rnn : latent -> latent
            decoder : latent -> output
        """
        super(autoencoderRNN, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_size, sum((input_size, hidden_size))//2),
                nn.Sigmoid(),
                nn.Linear(sum((input_size, hidden_size))//2, hidden_size),
                nn.Sigmoid(),
                )
        self.rnn = nn.RNN(
                hidden_size, 
                hidden_size, 
                rnn_layers, 
                batch_first=False, 
                bidirectional=bidirectional,
                dropout = dropout,
                )
        self.decoder = nn.Sequential(
                nn.Linear(hidden_size, sum((hidden_size, output_size))//2),
                nn.Sigmoid(),
                nn.Linear(sum((hidden_size, output_size))//2, output_size),
                nn.ReLU() if strictly_positive else nn.Identity()  # Use ReLU if strictly_positive is True
                )
        self.en_dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        out = self.encoder(x)
        out = self.en_dropout(out)
        latent_out, _ = self.rnn(out)
        out = self.decoder(latent_out)
        return out, latent_out


class LatentPoolCTRNN(nn.Module):
    """
    CTRNN with a large pool of latent neurons that can handle:
    - Multiple recording datasets with variable numbers of observed neurons
    - Multiple stimulus neurons indicating stimulus type
    - Continuous input for stimulus concentration
    - Estimation of smooth firing rates from binned spike counts
    
    Architecture:
        - Flexible encoder: maps variable-size neural observations to fixed latent space
        - Stimulus encoder: processes stimulus type and concentration
        - Large latent CTRNN: learns shared dynamics across datasets
        - Flexible decoder: maps latent space back to variable-size neural observations
    
    Parameters:
        max_observed_neurons: Maximum number of observed neurons across datasets
        latent_size: Number of latent neurons (should be large, e.g., 128-512)
        num_stimulus_types: Number of different stimulus types
        dt: Time step for CTRNN dynamics
        dropout: Dropout rate for regularization
    """
    
    def __init__(
            self,
            max_observed_neurons,
            latent_size=256,
            num_stimulus_types=4,
            dt=None,
            dropout=0.2,
            strictly_positive=True,
    ):
        super(LatentPoolCTRNN, self).__init__()
        
        self.max_observed_neurons = max_observed_neurons
        self.latent_size = latent_size
        self.num_stimulus_types = num_stimulus_types
        self.strictly_positive = strictly_positive
        
        # Encoder: maps observed neurons + stimulus info to latent space
        # Input: observed_neurons + num_stimulus_types (one-hot) + 1 (concentration)
        encoder_input_size = max_observed_neurons + num_stimulus_types + 1
        
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_size, (encoder_input_size + latent_size) // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear((encoder_input_size + latent_size) // 2, latent_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        
        # Large latent CTRNN
        self.ctrnn = CTRNN(
            input_size=latent_size,
            hidden_size=latent_size,
            dt=dt
        )
        
        # Decoder: maps latent space back to observed neurons
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, (latent_size + max_observed_neurons) // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear((latent_size + max_observed_neurons) // 2, max_observed_neurons),
            nn.ReLU() if strictly_positive else nn.Identity(),
        )
        
    def forward(self, spike_counts, stimulus_type, stimulus_concentration, mask=None):
        """
        Forward pass through the network.
        
        Args:
            spike_counts: (seq_len, batch, max_observed_neurons) - binned spike counts
                         Can be padded with zeros for datasets with fewer neurons
            stimulus_type: (seq_len, batch, num_stimulus_types) - one-hot encoded stimulus type
            stimulus_concentration: (seq_len, batch, 1) - continuous stimulus concentration
            mask: (batch, max_observed_neurons) - binary mask indicating which neurons are observed
                  1 for observed neurons, 0 for padding. If None, all neurons are considered observed.
        
        Returns:
            firing_rates: (seq_len, batch, max_observed_neurons) - estimated smooth firing rates
            latent_activity: (seq_len, batch, latent_size) - latent neuron activity
        """
        # Concatenate all inputs
        encoder_input = torch.cat([spike_counts, stimulus_type, stimulus_concentration], dim=-1)
        
        # Encode to latent space
        latent_input = self.encoder(encoder_input)
        
        # Process through CTRNN
        latent_activity, _ = self.ctrnn(latent_input)
        
        # Decode back to firing rates
        firing_rates = self.decoder(latent_activity)
        
        # Apply mask if provided (zero out non-observed neurons)
        if mask is not None:
            firing_rates = firing_rates * mask.unsqueeze(0)
        
        return firing_rates, latent_activity

