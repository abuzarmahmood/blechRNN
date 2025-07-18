import pytest
import torch
from src.train import train_model, smooth_MSELoss

def test_train_model():
    # Mock model, inputs, and labels for testing
    model = torch.nn.Linear(10, 5)
    inputs = torch.randn(5, 3, 10)
    labels = torch.randn(5, 3, 5)
    trained_model, loss_history, cross_val_loss = train_model(
        model, inputs, labels, output_size=5, train_steps=10
    )
    assert len(loss_history) > 0

def test_smooth_mse_loss():
    criterion = smooth_MSELoss()
    input_data = torch.randn(5, 3, 10)
    target_data = torch.randn(5, 3, 10)
    loss = criterion(input_data, target_data)
    assert loss.item() >= 0
