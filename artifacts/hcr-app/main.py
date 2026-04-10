import os
import base64
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "character_model.h5")

EMNIST_BALANCED_MAPPING = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

load_model()

def preprocess_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
        white_bg = np.ones_like(bgr) * 255
        alpha_factor = alpha[:, :, np.newaxis] / 255.0
        img = (bgr * alpha_factor + white_bg * (1 - alpha_factor)).astype(np.uint8)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)

    pad = 2
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + 2 * pad)
    h = min(gray.shape[0] - y, h + 2 * pad)

    cropped = binary[y:y+h, x:x+w]

    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)

    normalized = resized.astype(np.float32) / 255.0

    return normalized.reshape(1, 28, 28, 1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        if not load_model():
            return jsonify({
                'error': 'Model not found. Please run model_trainer.py first to train and save the model.',
                'model_missing': True
            }), 503

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image']

    processed = preprocess_image(image_data)
    if processed is None:
        return jsonify({'error': 'Canvas appears to be empty. Please draw a character first.'}), 400

    if np.max(processed) < 0.05:
        return jsonify({'error': 'Canvas appears to be empty. Please draw a character first.'}), 400

    try:
        predictions = model.predict(processed, verbose=0)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])
        character = EMNIST_BALANCED_MAPPING[class_idx]

        top5_indices = np.argsort(predictions[0])[::-1][:5]
        alternatives = [
            {
                'character': EMNIST_BALANCED_MAPPING[i],
                'confidence': float(predictions[0][i])
            }
            for i in top5_indices[1:]
        ]

        return jsonify({
            'character': character,
            'confidence': round(confidence * 100, 2),
            'alternatives': alternatives
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8008))
    app.run(host='0.0.0.0', port=port, debug=False)
