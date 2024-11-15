from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import io
import base64
from PIL import Image
import json
import time

from models.cnn import ConvolutionalNeuralNetwork
from models.snn import SimpleNeuralNetwork

app = Flask(__name__)
CORS(app)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables to store training state
current_training_state = {
    'model': None,
    'train_loader': None,
    'criterion': None,
    'optimizer': None,
    'epochs': 0,
    'dataset_name': '',
    'is_training': False
}

def load_dataset(dataset_name, batch_size):
    """Load and prepare dataset for training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_name in ['mnist', 'fashionmnist']:
        # MNIST and FashionMNIST are grayscale, so we use different transforms
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        # For CIFAR10 and Flowers102 (RGB images)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Select dataset based on name
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True,
                                     download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)
        input_channels = 1
        num_classes = 10
    elif dataset_name == 'fashionmnist':
        train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                           download=True, transform=transform)
        input_channels = 1
        num_classes = 10
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
        input_channels = 3
        num_classes = 10
    else:  # flowers102
        train_dataset = datasets.Flowers102(root='./data', split='train',
                                          download=True, transform=transform)
        test_dataset = datasets.Flowers102(root='./data', split='test',
                                         download=True, transform=transform)
        input_channels = 3
        num_classes = 102

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, input_channels, num_classes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global current_training_state
    try:
        data = request.json
        model_type = data['modelType']
        optimizer_type = data['optimizer']
        batch_size = int(data['batchSize'])
        epochs = int(data['epochs'])
        dataset = data['dataset']

        # Load dataset
        train_loader, _, input_channels, num_classes = load_dataset(dataset, batch_size)

        # Initialize model
        if model_type == 'snn':
            input_size = 28 if dataset in ['mnist', 'fashionmnist'] else 32
            model = SimpleNeuralNetwork(
                input_channels=input_channels,
                input_size=input_size,
                num_classes=num_classes
            ).to(device)
        else:
            kernels = [int(k) for k in data['kernels1'].split(',')]
            model = ConvolutionalNeuralNetwork(
                input_channels=input_channels,
                kernels=kernels,
                num_classes=num_classes
            ).to(device)

        # Setup criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters())
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Initialize training state
        current_training_state.update({
            'model': model,
            'train_loader': train_loader,
            'criterion': criterion,
            'optimizer': optimizer,
            'epochs': epochs,
            'dataset_name': dataset,
            'is_training': True  # Set this to True when training starts
        })

        return jsonify({'status': 'started'})
        
    except Exception as e:
        print(f"Error in /train: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stream')
def stream():
    def generate():
        if not current_training_state['is_training']:
            yield f"data: {json.dumps({'error': 'Training not started'})}\n\n"
            return

        try:
            model = current_training_state['model']
            train_loader = current_training_state['train_loader']
            criterion = current_training_state['criterion']
            optimizer = current_training_state['optimizer']
            epochs = current_training_state['epochs']
            
            # Send initial state
            initial_data = {
                'current_epoch': 0,
                'total_epochs': epochs,
                'current_batch': 0,
                'total_batches': len(train_loader),
                'loss': 0,
                'accuracy': 0,
                'image': '',
                'prediction': '',
                'confidence': 0
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    running_loss += loss.item()
                    
                    if batch_idx % 5 == 0:  # Update every 5 batches
                        # Get sample image and prediction
                        sample_image = data[0].cpu()
                        pred = outputs[0].argmax().item()
                        confidence = torch.nn.functional.softmax(outputs[0], dim=0)[pred].item() * 100
                        
                        # Convert tensor to base64 image
                        img_buf = io.BytesIO()
                        if sample_image.shape[0] == 1:
                            sample_image = sample_image.repeat(3, 1, 1)
                        transforms.ToPILImage()(sample_image).save(img_buf, format='JPEG')
                        img_str = base64.b64encode(img_buf.getvalue()).decode()
                        
                        # Prepare update data
                        update_data = {
                            'current_epoch': epoch + 1,
                            'total_epochs': epochs,
                            'current_batch': batch_idx + 1,
                            'total_batches': len(train_loader),
                            'loss': running_loss / (batch_idx + 1),
                            'accuracy': 100. * correct / total,
                            'image': img_str,
                            'prediction': str(pred),
                            'confidence': confidence
                        }
                        
                        yield f"data: {json.dumps(update_data)}\n\n"
                        time.sleep(0.1)  # Small delay
                        
        except Exception as e:
            print(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            current_training_state['is_training'] = False
            
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
