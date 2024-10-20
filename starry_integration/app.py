from flask import Flask, request, jsonify, send_from_directory, render_template
import time
import zlib
import requests  # Make sure to import requests for API calls
from openai import OpenAI
starry_key = "API_KEY"
app = Flask(__name__)

# Assuming animal images are stored in a directory named 'images'
IMAGES_DIR = 'images'

def format_size(size_bytes):
    """Convert bytes to a human-readable format (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} Bytes"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"

@app.route('/')
def index():
    return render_template('index.html'), 200

@app.route('/get_animal_image')
def get_animal_image():
    animal = request.args.get('animal')
    prompt = f"Imagine an image of {animal} in a random crop field with a picture taken in a golden ratio."

    url = "https://api.starryai.com/creations/"

    payload = {
        "model": "lyra",
        "aspectRatio": "square",
        "highResolution": False,
        "images": 1,
        "steps": 20,
        "initialImageMode": "color",
        "prompt": prompt
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-API-Key": starry_key
    }

    headers2 = {
        "accept": "application/json",
        "X-API-Key": starry_key
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        job_id = data['id']
        creation_pickup_link = f"https://api.starryai.com/creations/{job_id}"
        sleeptime = 5
        for i in range(4):  # Check for completion with increasing wait times
            sleeptime += i + 1
            response = requests.get(creation_pickup_link, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == "completed":
                    image_urls = data['images']
                    break
            else:
                break  # Exit if there's an error getting the status
            time.sleep(sleeptime)
    image_url = image_urls[0]['url']
    print(image_urls)
    print(image_url)


    try:
        return jsonify({'image_url': image_url})
    except:
        return jsonify({'error': 'Failed to generate image'}), 500

@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory(IMAGES_DIR, path)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_content = file.read()
        file_size = len(file_content)
        estimated_zip_size = len(zlib.compress(file_content))
        return jsonify({
            'name': file.filename,
            'size': format_size(file_size),  # Convert size to human-readable format
            'type': file.content_type,
            'estimated_zip_size': format_size(estimated_zip_size)  # Convert estimated size as well
        })

@app.route('/<path:path>', methods=['GET'])
def catch_all(path):
    return jsonify({'error': 'Not Found', 'path': path}), 404

if __name__ == '__main__':
    app.run(debug=True)
