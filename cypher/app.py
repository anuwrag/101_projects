from flask import Flask, render_template, jsonify
from model import MNISTModel
from train import train_model
import torch
import torch.nn as nn

app = Flask(__name__)

model = MNISTModel()

def get_layer_params(model):
    layer_params = []
    
    # First Conv Layer parameters
    conv1 = model.features[0]
    conv1_params = conv1.out_channels * (conv1.kernel_size[0] * conv1.kernel_size[1] * conv1.in_channels + 1)
    layer_params.append({
        'name': 'Conv Layer 1',
        'details': {
            'filters': conv1.out_channels,
            'kernel_size': f'{conv1.kernel_size[0]}x{conv1.kernel_size[1]}',
            'input_channels': conv1.in_channels,
            'params': conv1_params,
            'output_shape': '13x13x8 (after MaxPool)'
        }
    })
    
    # Second Conv Layer parameters
    conv2 = model.features[3]
    conv2_params = conv2.out_channels * (conv2.kernel_size[0] * conv2.kernel_size[1] * conv2.in_channels + 1)
    layer_params.append({
        'name': 'Conv Layer 2',
        'details': {
            'filters': conv2.out_channels,
            'kernel_size': f'{conv2.kernel_size[0]}x{conv2.kernel_size[1]}',
            'input_channels': conv2.in_channels,
            'params': conv2_params,
            'output_shape': '5x5x16 (after MaxPool)'
        }
    })
    
    # FC Layer parameters
    fc = model.classifier
    fc_params = fc.out_features * (fc.in_features + 1)
    layer_params.append({
        'name': 'FC Layer',
        'details': {
            'input_features': fc.in_features,
            'output_features': fc.out_features,
            'params': fc_params
        }
    })
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return layer_params, total_params

@app.route('/')
def index():
    layer_params, total_params = get_layer_params(model)
    return render_template('index.html', 
                         layer_params=layer_params,
                         total_params=total_params)

@app.route('/train')
def train():
    training_history = train_model(model)
    return jsonify(training_history)

if __name__ == '__main__':
    app.run(debug=True)
