import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_channels=1, input_size=28, hidden_size=512, num_classes=10):
        super(SimpleNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        total_input_size = input_channels * input_size * input_size
        
        self.layers = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x) 