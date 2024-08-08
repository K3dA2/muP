import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from model import VAE, VAEConfig
from utils import get_data_loader
from tqdm import tqdm

def kl_loss(mu, l_sigma):
    return -0.5 * torch.sum(1 + l_sigma - mu**2 - torch.exp(l_sigma))

def validate_model(model, val_loader, config, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, _ = data
            inputs = inputs.to(config.device)
            
            mu, l_sigma, outputs = model(inputs)
            mse = criterion(outputs, inputs)
            kl = kl_loss(mu, l_sigma)
            loss = mse + (kl * 0.0001)
            
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def train_model(model, train_loader, val_loader, config, num_epochs=20):
    criterion = nn.MSELoss()
    kl_weight = 0.0001
    layer_lrs = model.get_layer_lrs()
    optimizer = optim.AdamW(layer_lrs)
    model.to(config.device)
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as epoch_pbar:
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.to(config.device)
                
                optimizer.zero_grad()
                
                mu, l_sigma, outputs = model(inputs)
                kl = kl_loss(mu, l_sigma)
                mse = criterion(outputs, inputs)
                loss = mse + (kl * kl_weight)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                epoch_pbar.set_postfix({'loss': running_loss / (epoch_pbar.n + 1)})
                epoch_pbar.update(1)
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, config, criterion)
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
    
    return train_loss_history, val_loss_history

if __name__ == "__main__":
    path = '/Users/ayanfe/Documents/Datasets/Waifus/Train'
    val_path = '/Users/ayanfe/Documents/Datasets/Waifus/Val'

    train_loader = get_data_loader(path, batch_size=64, num_samples=20_000)
    val_loader = get_data_loader(val_path, batch_size=64, num_samples=10_000)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    learning_rates = [0.001, 0.0005, 3e-4, 0.0001]
    base_channel_sizes = [8, 16, 32]
    
    results = {}

    total_iterations = len(learning_rates) * len(base_channel_sizes)
    with tqdm(total=total_iterations, desc="Training models") as pbar:
        for lr in learning_rates:
            for base_channels in base_channel_sizes:
                print(f"Training with learning rate: {lr} and base channels: {base_channels}")
                
                config = VAEConfig(input_channels=3, z_dim=8, base_channels=base_channels, device=device, learning_rate=lr)
                model = VAE(config)
                
                train_loss_history, val_loss_history = train_model(model, train_loader, val_loader, config, num_epochs=10)
                results[(lr, base_channels)] = (train_loss_history, val_loss_history)
                pbar.update(1)
    
    # Plotting results
    plt.figure(figsize=(12, 8))
    for (lr, base_channels), (train_loss_history, val_loss_history) in results.items():
        plt.plot(np.log2(lr), val_loss_history, label=f'LR: {lr}, Base Channels: {base_channels}')
    
    plt.xlabel('Log2(Learning Rate)')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs. Log2(Learning Rate) for Different Base Channel Sizes')
    plt.legend()
    plt.grid(True)
    plt.show()
