from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
from nltk.corpus import wordnet
import nltk
import random
import os
from pathlib import Path
import torchaudio
import torch
import librosa
import librosa.display
import soundfile as sf
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues
import matplotlib.pyplot as plt
from scipy import signal

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

app = Flask(__name__)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Image transformations
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

def process_text(text):
    # Existing text processing code
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    
    # Random insertion
    words = text.split()
    insert_words = ['amazing', 'interesting', 'important', 'significant', 'notable']
    num_insertions = max(1, len(words) // 5)
    
    augmented_text1 = words.copy()
    for _ in range(num_insertions):
        insert_word = random.choice(insert_words)
        position = random.randint(0, len(augmented_text1))
        augmented_text1.insert(position, f'<span class="inserted">{insert_word}</span>')
    
    # Synonym replacement
    def get_synonyms(word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        return list(set(synonyms))
    
    augmented_text2 = words.copy()
    num_replacements = max(1, len(words) // 5)
    replaced_indices = random.sample(range(len(augmented_text2)), min(num_replacements, len(augmented_text2)))
    
    for idx in replaced_indices:
        word = augmented_text2[idx]
        synonyms = get_synonyms(word)
        if synonyms:
            replacement = random.choice(synonyms)
            augmented_text2[idx] = f'<span class="replaced">{replacement}</span>'
    
    return {
        'type': 'text',
        'tokens': [f'<span class="token">{token}</span>' for token in tokens],
        'token_count': token_count,
        'augmented_text1': ' '.join(augmented_text1),
        'augmented_text2': ' '.join(augmented_text2)
    }

def process_image(image_data):
    # Convert base64 to PIL Image
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define transformations
    transform_normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    transform_rotate = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])
    
    transform_brightness = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])
    
    # Apply transformations
    normalized_tensor = transform_normalize(image)
    rotated_tensor = transform_rotate(image)
    brightness_tensor = transform_brightness(image)
    
    # Convert tensors back to images
    def tensor_to_base64(tensor):
        img = transforms.ToPILImage()(tensor)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f'data:image/png;base64,{img_str}'
    
    return {
        'type': 'image',
        'normalized_image': tensor_to_base64(normalized_tensor),
        'rotated_image': tensor_to_base64(rotated_tensor),
        'brightness_image': tensor_to_base64(brightness_tensor),
        'norm_mean': normalized_tensor.mean().item(),
        'norm_std': normalized_tensor.std().item()
    }

def process_obj(obj_content):
    # Parse OBJ content
    vertices = []
    faces = []
    for line in obj_content.split('\n'):
        if line.startswith('v '):
            vertices.append([float(x) for x in line[2:].split()])
        elif line.startswith('f '):
            faces.append([int(x.split('/')[0]) for x in line[2:].split()])
    
    vertices = np.array(vertices)
    
    # Center the object
    center = vertices.mean(axis=0)
    centered_vertices = vertices - center
    
    # Random rotation
    angle = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    rotated_vertices = centered_vertices @ rotation_matrix
    
    # Scale the object
    scale_factor = np.random.uniform(0.5, 2.0)
    scaled_vertices = centered_vertices * scale_factor
    
    # Convert back to OBJ format
    def vertices_to_obj(verts, orig_faces):
        obj_lines = []
        for v in verts:
            obj_lines.append(f'v {v[0]} {v[1]} {v[2]}')
        for f in faces:
            obj_lines.append(f'f {" ".join(map(str, f))}')
        return '\n'.join(obj_lines)
    
    return {
        'type': 'obj',
        'centered_obj': vertices_to_obj(centered_vertices, faces),
        'rotated_obj': vertices_to_obj(rotated_vertices, faces),
        'scaled_obj': vertices_to_obj(scaled_vertices, faces)
    }

def process_audio(audio_data):
    try:
        # Convert base64 to audio
        audio_data = audio_data.split(',')[1]
        audio_bytes = base64.b64decode(audio_data)
        
        # Save temporarily
        temp_path = 'temp_audio.wav'
        with open(temp_path, 'wb') as f:
            f.write(audio_bytes)
        
        # Load and resample audio
        y, sr = librosa.load(temp_path, sr=16000)
        
        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Create MFCC plot
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC')
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        mfcc_plot = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        # Pitch shifting
        def pitch_shift(audio, steps):
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        
        high_pitch = pitch_shift(y, 4)  # Shift up by 4 semitones
        low_pitch = pitch_shift(y, -4)  # Shift down by 4 semitones
        
        # Add background noise (simulated car honk sound)
        t = np.linspace(0, len(y)/sr, len(y))
        honk_freq = 400  # Frequency for car honk sound
        noise = 0.1 * np.sin(2 * np.pi * honk_freq * t)
        noisy = y + noise
        
        # Function to convert audio to base64
        def audio_to_base64(audio_data):
            buf = io.BytesIO()
            sf.write(buf, audio_data, sr, format='wav')
            buf.seek(0)
            return f'data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode()}'
        
        # Clean up
        os.remove(temp_path)
        
        return {
            'type': 'audio',
            'sample_rate': sr,
            'mfcc_plot': f'data:image/png;base64,{mfcc_plot}',
            'resampled_audio': audio_to_base64(y),
            'high_pitch_audio': audio_to_base64(high_pitch),
            'low_pitch_audio': audio_to_base64(low_pitch),
            'noisy_audio': audio_to_base64(noisy)
        }
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return {'error': str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        request_type = request.form.get('type')
        
        if request_type == 'audio':
            if 'audio' not in request.form:
                return jsonify({'error': 'No audio file provided'})
            return jsonify(process_audio(request.form['audio']))
        elif request_type == 'obj':
            if 'obj' not in request.form:
                return jsonify({'error': 'No OBJ file provided'})
            return jsonify(process_obj(request.form['obj']))
        elif request_type == 'image':
            return jsonify(process_image(request.form['image']))
        elif request_type == 'text':
            return jsonify(process_text(request.form['text_content']))
        
        return jsonify({'error': 'Invalid request type'})
        
    except Exception as e:
        print(f"Error in process route: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)