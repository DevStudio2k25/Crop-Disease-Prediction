"""
Flask Web Application for Crop Disease Prediction
Provides a web interface for uploading images and getting predictions
"""

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/best_model.h5'
CLASS_INDICES_PATH = 'models/class_indices.json'
IMG_SIZE = 224

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and class indices
print("Loading model...")
model = load_model(MODEL_PATH)
print("✓ Model loaded successfully")

with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)
print("✓ Class indices loaded")

# Disease information - Simple and clean for demo
DISEASE_INFO = {
    'healthy': {
        'name': 'Healthy Plant',
        'status': 'No disease detected',
        'color': '#28a745'
    },
    'bacterial_spot': {
        'name': 'Bacterial Spot',
        'status': 'Disease detected',
        'color': '#dc3545'
    },
    'early_blight': {
        'name': 'Early Blight',
        'status': 'Disease detected',
        'color': '#fd7e14'
    },
    'late_blight': {
        'name': 'Late Blight',
        'status': 'Disease detected',
        'color': '#dc3545'
    },
    'leaf_mold': {
        'name': 'Leaf Mold',
        'status': 'Disease detected',
        'color': '#ffc107'
    },
    'septoria_leaf_spot': {
        'name': 'Septoria Leaf Spot',
        'status': 'Disease detected',
        'color': '#fd7e14'
    },
    'spider_mites': {
        'name': 'Spider Mites Infestation',
        'status': 'Pest detected',
        'color': '#ffc107'
    },
    'target_spot': {
        'name': 'Target Spot',
        'status': 'Disease detected',
        'color': '#fd7e14'
    },
    'mosaic_virus': {
        'name': 'Mosaic Virus',
        'status': 'Disease detected',
        'color': '#dc3545'
    },
    'yellow_leaf_curl': {
        'name': 'Yellow Leaf Curl Virus',
        'status': 'Disease detected',
        'color': '#dc3545'
    },
    'rust': {
        'name': 'Rust Disease',
        'status': 'Disease detected',
        'color': '#fd7e14'
    },
    'scab': {
        'name': 'Scab Disease',
        'status': 'Disease detected',
        'color': '#ffc107'
    },
    'multiple_diseases': {
        'name': 'Multiple Diseases',
        'status': 'Multiple diseases detected',
        'color': '#dc3545'
    }
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """
    Preprocess image for model prediction
    - Load image
    - Resize to model input size
    - Normalize pixel values
    - Add batch dimension
    """
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_disease(img_path):
    """
    Predict disease from image
    Returns: predicted class, confidence, and all probabilities
    """
    # Preprocess image
    processed_img = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)
    
    # Get predicted class
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = class_indices[str(predicted_class_idx)]
    
    # Get all class probabilities
    all_predictions = {}
    for idx, prob in enumerate(predictions[0]):
        class_name = class_indices[str(idx)]
        all_predictions[class_name] = float(prob)
    
    return predicted_class, confidence, all_predictions

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make prediction
            predicted_class, confidence, all_predictions = predict_disease(filepath)
            
            # Get disease information
            disease_data = DISEASE_INFO.get(predicted_class, {})
            
            # Prepare response
            response = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': round(confidence * 100, 2),
                'disease_info': disease_data,
                'all_predictions': {k: round(v * 100, 2) for k, v in all_predictions.items()},
                'image_path': filepath
            }
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG'}), 400

@app.route('/about')
def about():
    """Render about page"""
    return render_template('about.html')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("CROP DISEASE PREDICTION WEB APPLICATION")
    print("="*60)
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
