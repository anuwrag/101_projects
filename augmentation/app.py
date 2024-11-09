from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
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

def process_text(text):
    # Tokenization
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
        'tokens': [f'<span class="token">{token}</span>' for token in tokens],
        'token_count': token_count,
        'augmented_text1': ' '.join(augmented_text1),
        'augmented_text2': ' '.join(augmented_text2)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    print("Received process request")
    if 'text' not in request.files and 'text_content' not in request.form:
        print("No text provided")
        return jsonify({'error': 'No text provided'})
    
    if 'text_content' in request.form:
        text = request.form['text_content']
    else:
        text = request.files['text'].read().decode('utf-8')
    
    result = process_text(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)