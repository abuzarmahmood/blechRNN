import pytest
import torch
from src.model import CTRNN, CTRNN_plus_output, autoencoderRNN

def test_ctrnn_initialization():
    model = CTRNN(input_size=10, hidden_size=20)
    assert model.input_size == 10
    assert model.hidden_size == 20

def test_ctrnn_forward():
    model = CTRNN(input_size=10, hidden_size=20)
    input_data = torch.randn(5, 3, 10)  # seq_len, batch, input_size
    output, hidden = model(input_data)
    assert output.shape == (5, 3, 20)
    assert hidden.shape == (3, 20)

def test_autoencoder_rnn_forward():
    model = autoencoderRNN(input_size=10, hidden_size=20, output_size=5)
    input_data = torch.randn(5, 3, 10)  # seq_len, batch, input_size
    output, latent_out = model(input_data)
    assert output.shape == (5, 3, 5)
    assert latent_out.shape == (5, 3, 20)
