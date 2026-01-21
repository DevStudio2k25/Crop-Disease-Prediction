"""
Model Download Script
Downloads the trained model from cloud storage on deployment
"""

import os
import requests
from pathlib import Path

MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'best_model.h5'

# You need to upload your model to Google Drive, Dropbox, or similar
# and get a direct download link
MODEL_URL = "YOUR_MODEL_DOWNLOAD_URL_HERE"

def download_model():
    """Download model if not present"""
    
    if MODEL_PATH.exists():
        print("✓ Model already exists")
        return True
    
    print("Downloading model...")
    MODEL_DIR.mkdir(exist_ok=True)
    
    try:
        # For Google Drive links, you might need to use gdown library
        # pip install gdown
        # import gdown
        # gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)
        
        # For direct links:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("✓ Model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    download_model()
