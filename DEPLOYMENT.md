# Deployment Guide - Crop Disease Prediction

## 🚀 Quick Deployment Steps

### Problem: Model file is too large (232MB) for GitHub/Vercel

### ✅ Solution: Use Render.com or Hugging Face

---

## Option 1: Render.com (RECOMMENDED - Easiest)

### Step 1: Upload Model to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Upload `models/best_model.h5`
3. Right-click → Get link → Set to "Anyone with the link"
4. Copy the file ID from the link
   - Link format: `https://drive.google.com/file/d/FILE_ID_HERE/view`
5. Create direct download link:
   ```
   https://drive.google.com/uc?export=download&id=FILE_ID_HERE
   ```

### Step 2: Update app.py

Add this code at the top of `app.py` (after imports):

```python
import os
import gdown

MODEL_PATH = 'models/best_model.h5'

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    os.makedirs('models', exist_ok=True)
    # Replace with your Google Drive file ID
    file_id = "YOUR_FILE_ID_HERE"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("✓ Model downloaded!")
```

### Step 3: Add gdown to requirements.txt

```
gdown>=4.7.1
```

### Step 4: Deploy on Render

1. Go to [render.com](https://render.com)
2. Sign up / Login
3. Click "New +" → "Web Service"
4. Connect your GitHub repo
5. Settings:
   - **Name**: crop-disease-prediction
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free
6. Click "Create Web Service"
7. Wait 5-10 minutes for deployment
8. Your app will be live at: `https://crop-disease-prediction.onrender.com`

---

## Option 2: Hugging Face Spaces (For ML Projects)

### Step 1: Create Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up / Login

### Step 2: Create New Space
1. Click "New" → "Space"
2. Name: `crop-disease-prediction`
3. SDK: **Gradio** or **Streamlit**
4. Click "Create Space"

### Step 3: Upload Files
1. Upload all your code files
2. Upload `best_model.h5` (Hugging Face supports large files!)
3. Push to the Space

### Step 4: Done!
Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/crop-disease-prediction`

---

## Option 3: Railway.app (Alternative)

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Railway will auto-detect Python and deploy
6. Add environment variables if needed
7. Done!

---

## ⚠️ Important Notes

### For Vercel:
- ❌ **NOT RECOMMENDED** - Vercel has 250MB limit and doesn't support TensorFlow well
- Use Render or Hugging Face instead

### Model File:
- Size: ~232MB
- Cannot be stored in Git
- Must be downloaded during deployment or hosted separately

### Free Tier Limits:
- **Render**: 750 hours/month free
- **Hugging Face**: Unlimited for public spaces
- **Railway**: $5 free credit/month

---

## 🎯 Recommended: Render.com

**Why?**
- ✅ Free tier available
- ✅ Supports Python + TensorFlow
- ✅ Easy deployment
- ✅ Auto-deploys on git push
- ✅ Custom domains
- ✅ SSL included

**Just follow Option 1 above!** 🚀
