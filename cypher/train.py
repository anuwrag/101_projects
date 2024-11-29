import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from PIL import Image
import base64
import io

class AugmentationDemo:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.original_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_augmented_samples(self, num_samples=5):
        # Load MNIST dataset
        dataset = datasets.MNIST('./data', train=True, download=True)
        
        samples = []
        for i in range(num_samples):
            img, label = dataset[i]
            
            # Get original image
            orig_tensor = self.original_transform(img)
            
            # Get augmented versions
            aug_versions = [self.transform(img) for _ in range(3)]
            
            # Convert to viewable format
            orig_img = self._tensor_to_base64(orig_tensor)
            aug_imgs = [self._tensor_to_base64(aug) for aug in aug_versions]
            
            samples.append({
                'original': orig_img,
                'augmented': aug_imgs,
                'label': label
            })
            
        return samples

    def _tensor_to_base64(self, tensor):
        # Denormalize
        tensor = tensor * 0.3081 + 0.1307
        # Convert to PIL Image
        img = transforms.ToPILImage()(tensor)
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

def train_model(model, epochs=1, batch_size=128):
    # Data preparation with augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_history = []
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 99:
                accuracy = 100. * correct / total
                avg_loss = running_loss / 100
                training_history.append({
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'loss': avg_loss,
                    'accuracy': accuracy
                })
                running_loss = 0.0
                correct = 0
                total = 0

    return training_history 