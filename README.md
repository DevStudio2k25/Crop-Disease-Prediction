# Crop Disease Prediction 🌱

Deep Learning-based Crop Disease Detection System using Convolutional Neural Networks.

## 🎯 Features

- **13 Disease Classes**: Detects 13 different crop diseases and healthy plants
- **High Accuracy**: Trained on 32,000+ images
- **Web Interface**: Easy-to-use Flask web application
- **Real-time Predictions**: Instant disease detection from uploaded images

## 📊 Supported Diseases

1. Bacterial Spot
2. Early Blight
3. Healthy
4. Late Blight
5. Leaf Mold
6. Mosaic Virus
7. Multiple Diseases
8. Rust
9. Scab
10. Septoria Leaf Spot
11. Spider Mites
12. Target Spot
13. Yellow Leaf Curl

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/DevStudio2k25/Crop-Disease-Prediction.git
cd Crop-Disease-Prediction

# Install dependencies
pip install -r requirements.txt

# Train the model (optional - if you want to retrain)
python train_fast.py

# Run the web application
python app.py
```

Visit `http://localhost:5000` in your browser.

## 📦 Deployment

### Option 1: Render.com (Recommended)

1. Fork this repository
2. Create account on [Render.com](https://render.com)
3. Create new Web Service
4. Connect your GitHub repository
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `gunicorn app:app`
7. **Important**: Upload your trained model (`best_model.h5`) to cloud storage (Google Drive, Dropbox, etc.) and update the download link in the app

### Option 2: Hugging Face Spaces

1. Create account on [Hugging Face](https://huggingface.co)
2. Create new Space
3. Upload model file to the Space
4. Deploy!

### Option 3: Local Server

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

## 🎓 Model Training

### Fast Training (Recommended)
```bash
python train_fast.py
```
- Optimized for speed (~10-15 min per epoch)
- Uses 160x160 images
- Batch size: 64

### Standard Training
```bash
python train_incremental.py
```
- Higher quality (~50 min per epoch)
- Uses 224x224 images
- Batch size: 32

## 📁 Project Structure

```
Crop-Disease-Prediction/
├── app.py                  # Flask web application
├── train_fast.py          # Fast training script
├── train_incremental.py   # Incremental training script
├── requirements.txt       # Python dependencies
├── Procfile              # Deployment configuration
├── runtime.txt           # Python version
├── templates/            # HTML templates
│   ├── index.html
│   └── about.html
├── static/              # CSS, JS, images
│   ├── style.css
│   └── script.js
└── models/              # Trained models (not in git)
    ├── best_model.h5
    └── class_indices.json
```

## 🔧 Technical Details

- **Framework**: TensorFlow/Keras
- **Architecture**: Custom CNN with 4 convolutional blocks
- **Input Size**: 224x224 (standard) or 160x160 (fast)
- **Dataset**: 32,000+ crop disease images
- **Classes**: 13 (12 diseases + healthy)

## ⚠️ Important Notes

### Model File
The trained model file (`best_model.h5`) is ~232MB and cannot be stored in Git. You have two options:

1. **Train your own model**: Run `train_fast.py` or `train_incremental.py`
2. **Download pre-trained model**: Contact the repository owner for the model file

### For Deployment
Upload your trained model to:
- Google Drive
- Dropbox
- Hugging Face
- AWS S3

Then update the download link in `app.py` or use the `download_model.py` script.

## 📝 License

MIT License - feel free to use for educational purposes

## 👨‍💻 Author

DevStudio2k25

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Show your support

Give a ⭐️ if this project helped you!
