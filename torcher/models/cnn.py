import torch
import torch.nn as nn

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_channels=1, kernels=[16, 32, 64], num_classes=10):
        super(ConvolutionalNeuralNetwork, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, kernels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(kernels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(kernels[0], kernels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(kernels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(kernels[1], kernels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(kernels[2]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate final feature map size
        if input_channels == 1:  # MNIST/FashionMNIST
            feature_size = kernels[2] * 3 * 3
        else:  # CIFAR10/Flowers102
            feature_size = kernels[2] * 4 * 4
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x 