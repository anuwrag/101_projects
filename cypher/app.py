from flask import Flask, render_template, jsonify
from model import MNISTModel
from train import train_model, AugmentationDemo
import torch

app = Flask(__name__)

model = MNISTModel()
aug_demo = AugmentationDemo()

def get_layer_info(model):
    layer_info = []
    total_params = 0
    
    # Iterate through model layers
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            params = layer.out_channels * (layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels + 1)
            total_params += params
            
            layer_info.append({
                'name': 'Conv2d',
                'config': {
                    'input_channels': layer.in_channels,
                    'output_channels': layer.out_channels,
                    'kernel_size': f"{layer.kernel_size[0]}x{layer.kernel_size[1]}",
                    'output_shape': f"{28-layer.kernel_size[0]+1}x{28-layer.kernel_size[1]+1}x{layer.out_channels}",
                    'after_pool': f"{(28-layer.kernel_size[0]+1)//2}x{(28-layer.kernel_size[1]+1)//2}x{layer.out_channels}"
                },
                'params': params
            })
            
        elif isinstance(layer, torch.nn.Linear):
            params = layer.out_features * (layer.in_features + 1)
            total_params += params
            
            layer_info.append({
                'name': 'Linear',
                'config': {
                    'input_features': layer.in_features,
                    'output_features': layer.out_features
                },
                'params': params
            })
    
    return layer_info, total_params

@app.route('/')
def index():
    layer_info, total_params = get_layer_info(model)
    aug_samples = aug_demo.get_augmented_samples()
    print("Augmented samples:", len(aug_samples))  # Debug print
    print("First sample keys:", aug_samples[0].keys() if aug_samples else "No samples")  # Debug print
    return render_template('index.html', 
                         layer_info=layer_info,
                         total_params=total_params,
                         aug_samples=aug_samples)

@app.route('/train')
def train():
    try:
        print("Starting training...")  # Debug log
        training_history = train_model(model)
        print(f"Training completed. History length: {len(training_history)}")  # Debug log
        return jsonify(training_history)
    except Exception as e:
        print(f"Error during training: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
