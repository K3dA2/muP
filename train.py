import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from muP.model import VAE,VAEConfig
from utils import get_data_loader
from tqdm import tqdm

def kl_loss(mu,l_sigma):
    kl_loss = -0.5 * torch.sum(1 + l_sigma - mu**2 - torch.exp(l_sigma))
    return kl_loss

def train_model(model, train_loader, config, num_epochs=20):
    # Define loss function
    criterion = nn.MSELoss()
    kl_weight = 0.0001
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)
    
    # Move model to the configured device
    model.to(config.device)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as epoch_pbar:
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.to(config.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                mu, l_sigma, outputs = model(inputs)

                kl = kl_loss(mu, l_sigma)
                # Compute loss
                mse = criterion(outputs, inputs)
                loss = mse + (kl * kl_weight)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Accumulate loss
                running_loss += loss.item()
                epoch_pbar.set_postfix({'loss': running_loss / (epoch_pbar.n + 1)})
                epoch_pbar.update(1)
        
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}')
    
    return loss_history

def test_model(model, test_loader, config):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.to(config.device)
            mu, l_sigma, outputs = model(inputs)
            # Display the original and reconstructed images
            model.reconstruct(inputs)
            break  # Only visualize the first batch

if __name__ == "__main__":
    path = '/Users/ayanfe/Documents/Datasets/Waifus/Train'
    val_path = '/Users/ayanfe/Documents/Datasets/Waifus/Val'

    train_loader = get_data_loader(path, batch_size=64, num_samples=40_000)
    test_loader = get_data_loader(val_path, batch_size=64, num_samples=10_000)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    # Different configurations
    learning_rates = [0.0005, 3e-4 ,0.0001]
    base_channel_sizes = [32, 64, 128]
    
    results = {}

    total_iterations = len(learning_rates) * len(base_channel_sizes)
    with tqdm(total=total_iterations, desc="Training models") as pbar:
        for lr in learning_rates:
            for base_channels in base_channel_sizes:
                print(f"Training with learning rate: {lr} and base channels: {base_channels}")
                
                config = VAEConfig(input_channels=3, z_dim=8, base_channels=base_channels, device=device, learning_rate=lr)
                model = VAE(config)
                
                loss_history = train_model(model, train_loader, config, num_epochs=10)
                results[(lr, base_channels)] = loss_history
                test_model(model, test_loader, config)
                pbar.update(1)
    
    # Plotting results
    plt.figure(figsize=(12, 8))
    for (lr, base_channels), loss_history in results.items():
        plt.plot(loss_history, label=f'LR: {lr}, Base Channels: {base_channels}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Learning Rates and Base Channel Sizes')
    plt.legend()
    plt.grid(True)
    plt.show()
