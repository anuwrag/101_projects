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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    print("Received process request")
    
    # Check request type
    if 'type' not in request.form:
        return jsonify({'error': 'Request type not specified'})
    
    request_type = request.form['type']
    
    if request_type == 'image':
        if 'image' not in request.form:
            return jsonify({'error': 'No image provided'})
        return jsonify(process_image(request.form['image']))
    
    elif request_type == 'text':
        if 'text_content' not in request.form:
            return jsonify({'error': 'No text provided'})
        return jsonify(process_text(request.form['text_content']))
    
    return jsonify({'error': 'Invalid request type'})

if __name__ == '__main__':
    app.run(debug=True)