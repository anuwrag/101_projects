import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
from PIL import Image
import numpy as np
import base64
import io

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MNISTModel
from train import AugmentationDemo

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = MNISTModel()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, which exceeds the limit of 25000"

def test_model_with_augmentation():
    """Test model performance with augmented data"""
    torch.manual_seed(42)
    
    model = MNISTModel()
    
    # Define augmentation transforms
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load training data with augmentation
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Train for one epoch with augmented data
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Accuracy: {100. * correct / total:.2f}%')
    
    final_accuracy = 100. * correct / total
    print(f'Final Accuracy with Augmentation: {final_accuracy:.2f}%')
    assert final_accuracy > 95, f"Model accuracy with augmentation is {final_accuracy:.2f}%, which is below the required 95%"

def test_augmentation_works():
    """Test that augmentation is actually modifying the images"""
    aug_demo = AugmentationDemo()
    samples = aug_demo.get_augmented_samples(num_samples=1)
    
    # Verify we got augmented samples
    assert len(samples) > 0, "No augmented samples generated"
    assert 'augmented' in samples[0], "No augmented images in sample"
    assert len(samples[0]['augmented']) > 0, "No augmented images generated"
    
    # Convert base64 images back to arrays for comparison
    aug_images = samples[0]['augmented']
    aug_arrays = []
    
    for aug_img in aug_images:
        img_data = base64.b64decode(aug_img)
        img = Image.open(io.BytesIO(img_data))
        aug_arrays.append(np.array(img))
    
    # Verify augmented images are different from each other
    for i in range(len(aug_arrays)):
        for j in range(i + 1, len(aug_arrays)):
            assert not np.array_equal(aug_arrays[i], aug_arrays[j]), \
                "Augmented images should be different from each other"

if __name__ == '__main__':
    test_parameter_count()
    test_model_with_augmentation()
    test_augmentation_works()