# BlechRNN

Neural network pipeline to estimate neural firing rates from spike train data using continuous-time RNNs.

## Architecture
The model uses an autoencoder architecture with three main components:

- **Encoder**: Projects input spike data to latent space
  - Reduces dimensionality of spike train data
  - Learns efficient compressed representations
  - Implemented as a feedforward neural network

- **RNN**: Learns temporal dynamics in latent space 
  - Continuous-time RNN (CTRNN) architecture
  - Captures temporal dependencies in neural activity
  - Supports variable time steps via dt parameter
  - Optional bidirectional processing

- **Decoder**: Projects latent representations back to firing rate space
  - Reconstructs firing rate estimates from latent space
  - Matches input dimensionality
  - Includes rectification to ensure non-negative rates

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

### Quick Start
The main pipeline follows these steps:
1. Loads spike train data and bins it
2. Preprocesses using PCA and scaling
3. Splits data into train/test sets
4. Trains the model with cross-validation
5. Saves trained model and generates performance plots

### Detailed Example

```python
from get_data import load_spike_data
from model import CTRNN_plus_output
from train import train_model
import torch

# Load and preprocess data
spike_data = load_spike_data('path/to/data.h5')
input_size = spike_data.shape[-1]

# Create model
model = CTRNN_plus_output(
    input_size=input_size,
    hidden_size=64,
    output_size=input_size,
    dt=0.1
)

# Train model
trained_model = train_model(
    net=model,
    inputs=spike_data,
    labels=spike_data,  # For autoencoder
    output_size=input_size,
    train_steps=1000,
    lr=0.01
)

# Save model
torch.save(trained_model.state_dict(), 'trained_model.pth')
```

### Configuration Options

Key parameters that can be tuned:
- `hidden_size`: Number of hidden neurons in CTRNN
- `dt`: Time step for continuous-time dynamics
- `train_steps`: Number of training iterations
- `lr`: Learning rate for optimization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blechRNN.git
cd blechRNN
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
All dependencies are listed in requirements.txt:
- PyTorch (2.3.1) - Deep learning framework
- NumPy (2.0.0) - Numerical computing
- Scikit-learn (1.4.0) - Data preprocessing
- Matplotlib (3.8.2) - Visualization
- SciPy (1.14.0) - Scientific computing
- tqdm (4.66.1) - Progress bars
## Contributing

We welcome contributions to BlechRNN! Here's how you can help:

### Reporting Issues
- Use the GitHub issue tracker
- Include detailed description and steps to reproduce
- Attach relevant data samples if possible

### Development Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings for new functions/classes
- Comment complex algorithms
- Keep functions focused and modular

### Running Tests
```bash
python -m pytest tests/
```
