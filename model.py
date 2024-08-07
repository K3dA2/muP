import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from utils import ResNet, StridedResNet, ResNetTranspose
import matplotlib.pyplot as plt
import uuid
import os

class VAEConfig:
    def __init__(self, input_channels=3, z_dim=8, base_channels=64, device='mps', learning_rate=0.001):
        self.input_channels = input_channels
        self.z_dim = z_dim
        self.base_channels = base_channels
        self.device = device
        self.learning_rate = learning_rate

class Encoder(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.res = StridedResNet(config.input_channels, config.base_channels)
        self.res1 = StridedResNet(config.base_channels, config.base_channels * 2)
        self.res2 = StridedResNet(config.base_channels * 2, config.base_channels * 4)
        self.mu = nn.Conv2d(config.base_channels * 4, config.z_dim, kernel_size=3, padding=1)
        self.l_sigma = nn.Conv2d(config.base_channels * 4, config.z_dim, kernel_size=3, padding=1)
        
        # Initialize weights according to µP
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.weight.size(1) * m.weight.size(2) * m.weight.size(3)
                nn.init.normal_(m.weight, 0, 1 / math.sqrt(fan_in))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.res(x)
        x = self.res1(x)
        x = self.res2(x)
        mu = self.mu(x)
        l_sigma = self.l_sigma(x)
        
        # Clamping l_sigma for numerical stability
        #l_sigma_clamped = torch.clamp(l_sigma, min=-10, max=10)
        
        z = mu + torch.exp(l_sigma / 2) * torch.rand_like(l_sigma)
        
        return mu, l_sigma, z

class Decoder(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.z_dim = config.z_dim
        self.fres = ResNetTranspose(config.z_dim, config.base_channels // 2)
        self.res = ResNetTranspose(config.base_channels // 2, config.base_channels)
        self.res1 = ResNet(config.base_channels, config.base_channels * 2)
        self.res2 = ResNetTranspose(config.base_channels * 2, config.base_channels * 4)
        self.res3 = ResNet(config.base_channels * 4, config.base_channels)
        self.conv = nn.Conv2d(config.base_channels, config.input_channels, kernel_size=3, padding=1)

        # Initialize weights according to µP
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1 / math.sqrt(m.in_channels))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        z = self.fres(z)
        
        z = self.res(z)
        z = self.res1(z)
        z = self.res2(z)
        z = self.res3(z)
        z = self.conv(z)
        
        return z

class VAE(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.device = config.device
        self.z_dim = config.z_dim
        self.learning_rate = config.learning_rate / math.sqrt(config.base_channels)

    def forward(self, x):
        mu, l_sigma, z = self.encoder.forward(x)
        out = self.decoder.forward(z)
        return mu, l_sigma, out
    
    def inferenceR(self, should_save=True):
        z_var = torch.rand(1, self.z_dim, 4, 4).to(self.device)
        self.decoder.eval()
        pred = self.decoder.forward(z_var)
        if should_save:
            plt.imshow(np.transpose(pred[-1].cpu().detach().numpy(), (1, 2, 0)))
            plt.axis('off')
            random_filename = str(uuid.uuid4()) + '.png'
            save_directory = 'Images/'
            full_path = os.path.join(save_directory, random_filename)
            plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.imshow(np.transpose(pred[-1].cpu().detach().numpy(), (1, 2, 0)))
            plt.show()
        self.decoder.train()

    def reconstruct(self, data):
        self.decoder.eval()
        self.encoder.eval()
        _, _, l_img = self.encoder(data)
        img_rec = self.decoder(l_img)
        img_orig = np.transpose(data[-1].cpu().detach().numpy(), (1, 2, 0))
        img_rec = np.transpose(img_rec[-1].cpu().detach().numpy(), (1, 2, 0))
        mean=[0.7002, 0.6099, 0.6036]
        std=[0.2195, 0.2234, 0.2097]
        mean = np.array(mean)
        std = np.array(std)
        img_rec = img_rec * std + mean
        img_rec = np.clip(img_rec, 0, 1)
        img_rec = (img_rec * 255).astype(np.uint8)
        img_concat = np.concatenate((img_orig, img_rec), axis=1)
        plt.imshow(img_concat)
        plt.axis('off')
        random_filename = str(uuid.uuid4()) + '.png'
        save_directory = 'Reconstructed/'
        os.makedirs(save_directory, exist_ok=True)
        full_path = os.path.join(save_directory, random_filename)
        plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
        self.decoder.train()
        self.encoder.train()

class TestEncoder(unittest.TestCase):
    def test_forward(self):
        '''
        model = Encoder(VAEConfig())
        input_tensor = torch.randn(1, 3, 64, 64)
        mu, sigma, z = model.forward(input_tensor)
        self.assertEqual(mu.shape, (1, 64))
        '''
        model = Decoder(VAEConfig())
        input_tensor = torch.randn(1, 4, 4, 4)
        out = model.forward(input_tensor)
        self.assertEqual(out.shape, (1, 3, 32, 32))

if __name__ == "__main__":
    unittest.main()
