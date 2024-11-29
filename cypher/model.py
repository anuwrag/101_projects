import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),  # 26x26x8
            nn.ReLU(),
            nn.MaxPool2d(2),  # 13x13x8
            
            nn.Conv2d(8, 16, kernel_size=3),  # 11x11x16
            nn.ReLU(),
            nn.MaxPool2d(2)  # 5x5x16
        )
        self.classifier = nn.Linear(5 * 5 * 16, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 5 * 5 * 16)
        x = self.classifier(x)
        return x
