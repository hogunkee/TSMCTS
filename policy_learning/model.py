import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PlaceNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2*hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Conv2d(4*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Conv2d(4*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4*hidden_dim, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        h = self.cnn(x)
        h = F.interpolate(h, scale_factor=1/4, mode='bilinear')
        _, C, H, W = h.shape
        h = h.reshape(-1, C*H*W)
        p = F.softmax(h, dim=1)
        prob = p.view(-1, H, W)
        return prob

