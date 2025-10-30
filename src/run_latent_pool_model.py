"""
Example script demonstrating how to use the LatentPoolCTRNN model
for training on multiple datasets with variable neuron counts.

This model addresses the following requirements:
1. Large pool of latent neurons that learn shared dynamics
2. Handles multiple recording datasets with variable numbers of observed neurons
3. Incorporates stimulus type (one-hot encoded) and concentration (continuous)
4. Estimates smooth firing rates from binned spike counts
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(__file__))
from model import LatentPoolCTRNN
from train import train_latent_pool_model, smooth_MSELoss


def prepare_dataset(
    spike_counts,
    stimulus_types,
    stimulus_concentrations,
    max_neurons,
    num_stimulus_types=4,
    bin_size=25
):
    """
    Prepare a single dataset for training with the LatentPoolCTRNN model.
    
    Args:
        spike_counts: (seq_len, batch, num_neurons) - binned spike counts
        stimulus_types: (batch,) - integer array of stimulus type indices
        stimulus_concentrations: (batch,) - continuous concentration values
        max_neurons: Maximum number of neurons across all datasets
        num_stimulus_types: Number of different stimulus types
        bin_size: Bin size in ms for spike counts
    
    Returns:
        Dictionary containing prepared data for training
    """
    seq_len, batch_size, num_neurons = spike_counts.shape
    
    # Pad spike counts to max_neurons if needed
    if num_neurons < max_neurons:
        padding = np.zeros((seq_len, batch_size, max_neurons - num_neurons))
        spike_counts_padded = np.concatenate([spike_counts, padding], axis=-1)
    else:
        spike_counts_padded = spike_counts
    
    # Create mask for observed neurons
    mask = np.zeros((batch_size, max_neurons))
    mask[:, :num_neurons] = 1
    
    # One-hot encode stimulus types
    stimulus_type_onehot = np.zeros((seq_len, batch_size, num_stimulus_types))
    for t in range(seq_len):
        for b in range(batch_size):
            stimulus_type_onehot[t, b, stimulus_types[b]] = 1
    
    # Expand stimulus concentration to match sequence length
    stimulus_conc_expanded = np.tile(
        stimulus_concentrations[np.newaxis, :, np.newaxis],
        (seq_len, 1, 1)
    )
    
    # Target rates: smooth version of spike counts (simple moving average)
    # In practice, you might use more sophisticated smoothing
    window_size = 3
    target_rates = np.copy(spike_counts_padded)
    for i in range(window_size // 2, seq_len - window_size // 2):
        target_rates[i] = np.mean(
            spike_counts_padded[i - window_size // 2:i + window_size // 2 + 1],
            axis=0
        )
    
    return {
        'spike_counts': torch.from_numpy(spike_counts_padded).float(),
        'stimulus_type': torch.from_numpy(stimulus_type_onehot).float(),
        'stimulus_concentration': torch.from_numpy(stimulus_conc_expanded).float(),
        'target_rates': torch.from_numpy(target_rates).float(),
        'mask': torch.from_numpy(mask).float()
    }


def create_synthetic_data(
    num_datasets=3,
    neurons_per_dataset=[30, 25, 35],
    num_trials=100,
    seq_len=200,
    num_stimulus_types=4
):
    """
    Create synthetic data for demonstration purposes.
    
    Args:
        num_datasets: Number of different recording datasets
        neurons_per_dataset: List of neuron counts for each dataset
        num_trials: Number of trials per dataset
        seq_len: Sequence length (time bins)
        num_stimulus_types: Number of stimulus types
    
    Returns:
        train_data_list: List of training data dictionaries
        test_data_list: List of test data dictionaries
    """
    max_neurons = max(neurons_per_dataset)
    train_data_list = []
    test_data_list = []
    
    for dataset_idx in range(num_datasets):
        num_neurons = neurons_per_dataset[dataset_idx]
        
        # Generate synthetic spike counts (Poisson-like)
        base_rate = 5 + np.random.randn(num_neurons) * 2
        base_rate = np.maximum(base_rate, 0.5)
        
        # Training data
        train_trials = int(0.8 * num_trials)
        train_spike_counts = np.random.poisson(
            base_rate[np.newaxis, np.newaxis, :],
            size=(seq_len, train_trials, num_neurons)
        ).astype(float)
        
        # Add stimulus-dependent modulation
        train_stimulus_types = np.random.randint(0, num_stimulus_types, train_trials)
        train_stimulus_conc = np.random.uniform(0.1, 1.0, train_trials)
        
        for trial in range(train_trials):
            stim_type = train_stimulus_types[trial]
            stim_conc = train_stimulus_conc[trial]
            # Modulate firing rates based on stimulus
            modulation = 1.0 + stim_conc * (stim_type / num_stimulus_types)
            train_spike_counts[:, trial, :] *= modulation
        
        train_data = prepare_dataset(
            train_spike_counts,
            train_stimulus_types,
            train_stimulus_conc,
            max_neurons,
            num_stimulus_types
        )
        train_data_list.append(train_data)
        
        # Test data
        test_trials = num_trials - train_trials
        test_spike_counts = np.random.poisson(
            base_rate[np.newaxis, np.newaxis, :],
            size=(seq_len, test_trials, num_neurons)
        ).astype(float)
        
        test_stimulus_types = np.random.randint(0, num_stimulus_types, test_trials)
        test_stimulus_conc = np.random.uniform(0.1, 1.0, test_trials)
        
        for trial in range(test_trials):
            stim_type = test_stimulus_types[trial]
            stim_conc = test_stimulus_conc[trial]
            modulation = 1.0 + stim_conc * (stim_type / num_stimulus_types)
            test_spike_counts[:, trial, :] *= modulation
        
        test_data = prepare_dataset(
            test_spike_counts,
            test_stimulus_types,
            test_stimulus_conc,
            max_neurons,
            num_stimulus_types
        )
        test_data_list.append(test_data)
    
    return train_data_list, test_data_list


def main():
    """
    Main function demonstrating the LatentPoolCTRNN model usage.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directories
    base_dir = os.path.dirname(os.path.dirname(__file__))
    plot_dir = os.path.join(base_dir, 'plots')
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print("=" * 60)
    print("LatentPoolCTRNN Model Demonstration")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    num_datasets = 3
    neurons_per_dataset = [30, 25, 35]
    max_neurons = max(neurons_per_dataset)
    num_stimulus_types = 4
    
    train_data_list, test_data_list = create_synthetic_data(
        num_datasets=num_datasets,
        neurons_per_dataset=neurons_per_dataset,
        num_trials=100,
        seq_len=200,
        num_stimulus_types=num_stimulus_types
    )
    
    print(f"   - Created {num_datasets} datasets")
    print(f"   - Neurons per dataset: {neurons_per_dataset}")
    print(f"   - Max neurons: {max_neurons}")
    print(f"   - Stimulus types: {num_stimulus_types}")
    
    # Create model
    print("\n2. Creating LatentPoolCTRNN model...")
    latent_size = 128  # Large pool of latent neurons
    model = LatentPoolCTRNN(
        max_observed_neurons=max_neurons,
        latent_size=latent_size,
        num_stimulus_types=num_stimulus_types,
        dt=10,  # 10ms time step
        dropout=0.2
    )
    
    print(f"   - Latent neurons: {latent_size}")
    print(f"   - Max observed neurons: {max_neurons}")
    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n3. Training model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"   - Using device: {device}")
    
    save_path = os.path.join(artifacts_dir, 'best_latent_pool_model.pt')
    
    model, loss_history, cross_val_loss = train_latent_pool_model(
        net=model,
        train_data_list=train_data_list,
        test_data_list=test_data_list,
        train_steps=2000,
        lr=0.001,
        device=device,
        criterion=smooth_MSELoss(alpha=0.05),
        save_path=save_path,
        patience=15
    )
    
    # Save final model
    final_model_path = os.path.join(artifacts_dir, 'final_latent_pool_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n4. Model saved to: {final_model_path}")
    
    # Plot training curves
    print("\n5. Generating plots...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_history, label='Training Loss', alpha=0.7)
    if cross_val_loss:
        steps = list(cross_val_loss.keys())
        losses = list(cross_val_loss.values())
        ax.plot(steps, losses, label='Validation Loss', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('LatentPoolCTRNN Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(plot_dir, 'latent_pool_training.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   - Training plot saved to: {plot_path}")
    plt.close(fig)
    
    # Evaluate model on test data
    print("\n6. Evaluating model on test data...")
    model.eval()
    with torch.no_grad():
        for dataset_idx, test_data in enumerate(test_data_list):
            spike_counts = test_data['spike_counts'].to(device)
            stimulus_type = test_data['stimulus_type'].to(device)
            stimulus_concentration = test_data['stimulus_concentration'].to(device)
            target_rates = test_data['target_rates'].to(device)
            mask = test_data['mask'].to(device)
            
            predicted_rates, latent_activity = model(
                spike_counts,
                stimulus_type,
                stimulus_concentration,
                mask
            )
            
            # Compute R² score for observed neurons
            mask_np = mask.cpu().numpy()
            pred_np = predicted_rates.cpu().numpy()
            target_np = target_rates.cpu().numpy()
            
            # Flatten and mask
            pred_flat = pred_np.reshape(-1, max_neurons)
            target_flat = target_np.reshape(-1, max_neurons)
            
            r2_scores = []
            for neuron_idx in range(max_neurons):
                if mask_np[0, neuron_idx] > 0:  # Only evaluate observed neurons
                    from sklearn.metrics import r2_score
                    r2 = r2_score(target_flat[:, neuron_idx], pred_flat[:, neuron_idx])
                    r2_scores.append(r2)
            
            mean_r2 = np.mean(r2_scores)
            print(f"   - Dataset {dataset_idx + 1}: Mean R² = {mean_r2:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
