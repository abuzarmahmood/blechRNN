# BlechRNN

Neural network pipeline to estimate neural firing rates from spike train data using continuous-time RNNs.

## Architecture
The model uses an autoencoder architecture with:
- Encoder: Projects input spike data to latent space
- RNN: Learns temporal dynamics in latent space 
- Decoder: Projects latent representations back to firing rate space

## Key Features
- Continuous-time RNN implementation
- Optional dropout for regularization
- PCA preprocessing of spike data
- MSE and Poisson loss functions
- Cross-validation during training
- GPU acceleration support

## Modules
- `get_data.py`: Loads and preprocesses spike train data from .h5 files
- `model.py`: Implements CTRNN and autoencoder network architectures
- `train.py`: Handles model training with cross-validation
- `run_model.py`: End-to-end pipeline for data loading, training and evaluation

## Usage
The main pipeline:
1. Loads spike train data and bins it
2. Preprocesses using PCA and scaling
3. Splits data into train/test sets
4. Trains the model with cross-validation
5. Saves trained model and generates performance plots

## Requirements
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
