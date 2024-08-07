import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset,Dataset
import random
import os
from PIL import Image

'''
This ResNet class is intended to be used as the smallest unit of the block class
'''

class ResNet(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32, useMaxPool = False, upscale = False,dropout_rate = 0.2):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.useMaxPool = useMaxPool
        self.upscale = upscale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        # Apply LayerNorm after conv1
        out = F.leaky_relu(self.conv1(x))
        out = self.dropout(out)
        out = F.layer_norm(out, out.size()[1:])
        
        # Apply LayerNorm after conv2
        out = F.leaky_relu(self.conv2(out))
        out = self.dropout(out)
        out = F.layer_norm(out, out.size()[1:])
        
        out1 = self.skip_conv(x)
        out = F.leaky_relu(self.conv3(out))
        
        if self.useMaxPool:
            #skip = out + out1
            out = F.max_pool2d(out + out1, 2)
            return out
        elif self.upscale:
            out = F.upsample(out + out1, scale_factor=2)
        else:
            out = F.leaky_relu(out + out1)
        return out
    
class StridedResNet(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32, stride = 2,dropout_rate = 0.2):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1,stride=stride)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=stride)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        # Apply LayerNorm after conv1
        out = F.leaky_relu(self.conv1(x))
        out = self.dropout(out)
        out = F.group_norm(out,4)
        
        # Apply LayerNorm after conv2
        out = F.leaky_relu(self.conv2(out))
        out = self.dropout(out)
        out = F.group_norm(out,4)
        
        out1 = self.skip_conv(x)
        out = F.leaky_relu(self.conv3(out))
        
        out = out + out1
        return out

class ResNetTranspose(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32, stride = 2,dropout_rate = 0.2):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.conv3 = nn.ConvTranspose2d(out_channels//2, out_channels, kernel_size=3, padding=1,stride=stride,output_padding=1)
        self.skip_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1,stride=stride,output_padding=1)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        # Apply LayerNorm after conv1
        out = F.leaky_relu(self.conv1(x))
        out = self.dropout(out)
        out = F.group_norm(out,4)
        
        # Apply LayerNorm after conv2
        out = F.leaky_relu(self.conv2(out))
        out = self.dropout(out)
        out = F.group_norm(out,4)
        
        out1 = self.skip_conv(x)
        out = F.leaky_relu(self.conv3(out))
        
        out = out + out1
        return out

class CustomDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label as it's not used
        
def get_data_loader(path, batch_size, num_samples=None, shuffle=True):
    # Define your transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.7002, 0.6099, 0.6036), (0.2195, 0.2234, 0.2097))  # Adjust these values if you have RGB images
    ])
    
    # Get the list of all image files in the root directory, excluding non-image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_extensions)]
    
    if len(image_files) == 0:
        raise ValueError("No valid image files found in the specified directory.")

    # If num_samples is not specified, use the entire dataset
    if num_samples is None or num_samples > len(image_files):
        num_samples = len(image_files)
    elif num_samples <= 0:
        raise ValueError("num_samples should be a positive integer.")

    print("data length: ", len(image_files))
    
    # Generate a list of indices to sample from (ensure dataset size is not exceeded)
    if shuffle:
        indices = random.sample(range(len(image_files)), num_samples)
    else:
        indices = list(range(num_samples))
    
    # Create the subset dataset
    subset_dataset = CustomDataset([image_files[i] for i in indices], transform=transform)
    
    # Create a DataLoader for the subset
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader

'''
Unit testing class
'''
class TestResNet(unittest.TestCase):
    def test_forward(self):
        model = StridedResNet(in_channels=16,out_channels = 16,stride=2)
        input_tensor = torch.randn(1, 16, 64, 64)  # Example input with shape (batch_size, channels, height, width)
        output = model.forward(input_tensor)
        #print(output.shape)
        self.assertEqual(output.shape, (1, 16, 64, 64))  # Adjust the expected shape based on your model architecture
        
        
if __name__ == '__main__':
    unittest.main()