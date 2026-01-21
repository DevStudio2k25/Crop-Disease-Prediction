# Crop Disease Prediction System

An AI-powered web application for detecting plant diseases using Convolutional Neural Networks (CNN). This system analyzes plant leaf images and provides instant disease diagnosis with treatment recommendations.

## 🌟 Features

- **Real-time Disease Detection**: Upload plant images and get instant predictions
- **High Accuracy**: Deep learning model trained on comprehensive plant disease dataset
- **Detailed Analysis**: Confidence scores and probability distribution for all disease classes
- **Treatment Recommendations**: Actionable advice for disease management
- **User-Friendly Interface**: Clean and intuitive web interface
- **Responsive Design**: Works on desktop and mobile devices

## 🎯 Disease Categories

The system can identify the following conditions:

1. **Healthy** - No disease detected
2. **Rust** - Fungal disease with orange-brown pustules
3. **Scab** - Dark scabby lesions on leaves and fruits
4. **Multiple Diseases** - Multiple infections present simultaneously

## 🏗️ Project Structure

```
crop-disease-prediction/
│
├── dataset/                    # Organized dataset (created by organize_data.py)
│   ├── train/
│   │   ├── healthy/
│   │   ├── multiple_diseases/
│   │   ├── rust/
│   │   └── scab/
│   └── test/
│
├── models/                     # Trained models (created during training)
│   ├── best_model.h5
│   ├── final_model.h5
│   ├── class_indices.json
│   └── training_history.png
│
├── static/                     # Static files for web app
│   ├── style.css
│   ├── script.js
│   └── uploads/
│
├── templates/                  # HTML templates
│   ├── index.html
│   └── about.html
│
├── organize_data.py           # Data organization script
├── train_model.py             # Model training script
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Installation & Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Organize Dataset

```bash
python organize_data.py
```

This script will:
- Read the CSV files containing labels
- Organize images into class-wise folders
- Create train/test split
- Display dataset summary

### Step 3: Train the Model

```bash
python train_model.py
```

This will:
- Build the CNN architecture
- Train the model with data augmentation
- Save the best model based on validation accuracy
- Generate training history plots
- Save class indices for prediction

Training parameters:
- Image size: 224x224
- Batch size: 32
- Epochs: 50 (with early stopping)
- Optimizer: Adam
- Loss: Categorical Crossentropy

### Step 4: Run Web Application

```bash
python app.py
```

Open your browser and navigate to: `http://localhost:5000`

## 🧠 Model Architecture

The CNN model consists of:

```
Input Layer (224x224x3)
    ↓
Conv2D (32 filters) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D (64 filters) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D (128 filters) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D (256 filters) → BatchNorm → MaxPool → Dropout
    ↓
Flatten
    ↓
Dense (512 units) → BatchNorm → Dropout
    ↓
Dense (256 units) → BatchNorm → Dropout
    ↓
Output Layer (4 units, Softmax)
```

**Key Features:**
- Batch Normalization for stable training
- Dropout layers for regularization
- MaxPooling for spatial dimension reduction
- ReLU activation for non-linearity
- Softmax for multi-class classification

## 📊 Training Process

1. **Data Preprocessing**:
   - Images resized to 224x224
   - Pixel values normalized to [0, 1]
   - 80-20 train-validation split

2. **Data Augmentation**:
   - Random rotation (±20°)
   - Width/height shift (20%)
   - Horizontal flip
   - Zoom (20%)
   - Shear transformation

3. **Training Strategy**:
   - Early stopping (patience: 10 epochs)
   - Learning rate reduction on plateau
   - Model checkpoint to save best weights

4. **Evaluation**:
   - Validation accuracy monitoring
   - Training/validation loss curves
   - Confusion matrix analysis

## 💻 Web Application Usage

1. **Upload Image**:
   - Click on upload box or drag & drop
   - Supported formats: JPG, JPEG, PNG
   - Maximum size: 16MB

2. **Get Prediction**:
   - Click "Analyze Image" button
   - Wait for processing (few seconds)
   - View detailed results

3. **Results Display**:
   - Disease name and confidence score
   - Description of the condition
   - Severity level
   - Treatment recommendations
   - Probability distribution for all classes

## 🔧 Technical Details

### Backend (Flask)
- Image upload handling
- Model loading and prediction
- JSON API responses
- Static file serving

### Frontend
- Responsive HTML/CSS design
- JavaScript for interactivity
- Drag-and-drop file upload
- Real-time result display
- Smooth animations

### Machine Learning
- TensorFlow/Keras framework
- Custom CNN architecture
- Image preprocessing pipeline
- Multi-class classification

## 📈 Model Performance

The model achieves high accuracy on the validation set. Performance metrics include:
- Training accuracy
- Validation accuracy
- Loss curves
- Per-class precision and recall

Detailed metrics are saved in `models/training_history.png`

## 🎓 Educational Value

This project demonstrates:
- End-to-end machine learning pipeline
- Deep learning for computer vision
- Web application development
- Model deployment
- User interface design

Perfect for:
- Final year projects
- Machine learning portfolios
- Agricultural technology demonstrations
- AI application showcases

## ⚠️ Important Notes

1. **Model Training**: Training may take 1-2 hours depending on hardware
2. **GPU Recommended**: For faster training, use GPU-enabled TensorFlow
3. **Dataset Size**: Ensure sufficient disk space for images
4. **Memory**: At least 8GB RAM recommended for training

## 🔒 Disclaimer

This system is designed for educational and assistive purposes. While it provides accurate predictions, it should not replace professional agricultural consultation. Always consult with agricultural experts for critical decisions regarding crop management.

## 📝 License

This project is created for educational purposes.

## 👨‍💻 Development

Built with:
- Python 3.8+
- TensorFlow 2.13
- Flask 2.3
- HTML5/CSS3/JavaScript

## 🤝 Contributing

This is an educational project. Feel free to fork and modify for your learning purposes.

## 📧 Support

For questions or issues, please refer to the code comments and documentation.

---

**Note**: This is a complete, standalone project suitable for academic presentations and demonstrations.
