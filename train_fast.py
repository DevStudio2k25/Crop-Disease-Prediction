"""
FAST Training Script - Quick training for deployment
Optimized for speed while maintaining good accuracy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
from pathlib import Path

# FAST Configuration - Optimized for speed
IMG_SIZE = 160  # Smaller = 2x faster (was 224)
BATCH_SIZE = 64  # Larger = 2x faster (was 32)
EPOCHS_PER_SESSION = 1  # 1 epoch at a time
NUM_CLASSES = 13

# Paths
MODEL_DIR = Path('models')
CHECKPOINT_PATH = MODEL_DIR / 'checkpoint_model.h5'
PROGRESS_FILE = MODEL_DIR / 'training_progress.json'
CLASS_INDICES_FILE = MODEL_DIR / 'class_indices.json'

def load_training_progress():
    """Load previous training progress"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        'total_epochs_completed': 0,
        'best_val_accuracy': 0.0,
        'history': {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }
    }

def save_training_progress(progress):
    """Save training progress"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=4)

def build_fast_model():
    """Build FASTER model - optimized architecture"""
    model = Sequential([
        # First Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Block - Smaller
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense Layers - Smaller
        Flatten(),
        Dense(256, activation='relu'),  # Reduced from 512
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu'),  # Reduced from 256
        Dropout(0.5),
        
        # Output
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_fast_data_generators():
    """Prepare data generators with less augmentation for speed"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,  # Reduced from 20
        width_shift_range=0.15,  # Reduced from 0.2
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,  # Reduced from 0.2
        fill_mode='nearest',
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        'complete_dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        'complete_dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def save_class_indices(train_generator):
    """Save class indices"""
    class_indices = train_generator.class_indices
    indices_to_class = {v: k for k, v in class_indices.items()}
    
    with open(CLASS_INDICES_FILE, 'w') as f:
        json.dump(indices_to_class, f, indent=4)

def plot_training_history(progress):
    """Plot training history"""
    history = progress['history']
    
    if not history['accuracy']:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['accuracy'], label='Training Accuracy')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['loss'], label='Training Loss')
    axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main FAST training pipeline"""
    
    print("="*70)
    print("⚡ FAST TRAINING MODE - Optimized for Speed!")
    print("="*70)
    print("🚀 Optimizations:")
    print("   • Image Size: 160x160 (2x faster)")
    print("   • Batch Size: 64 (2x faster)")
    print("   • Smaller Model (1.5x faster)")
    print("   • Less Augmentation (1.3x faster)")
    print("   📊 Total Speed: ~8x FASTER!")
    print("="*70)
    
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load progress
    progress = load_training_progress()
    total_epochs_done = progress['total_epochs_completed']
    
    print(f"\n📊 Previous Progress:")
    print(f"   ✓ Total epochs completed: {total_epochs_done}")
    print(f"   ✓ Best validation accuracy: {progress['best_val_accuracy']:.4f}")
    
    # Prepare data
    print("\n[1/4] Preparing data generators...")
    train_gen, val_gen = prepare_fast_data_generators()
    save_class_indices(train_gen)
    print(f"   ✓ Training samples: {train_gen.samples}")
    print(f"   ✓ Validation samples: {val_gen.samples}")
    print(f"   ✓ Steps per epoch: {len(train_gen)} (MUCH FASTER!)")
    
    # Load or build model
    print("\n[2/4] Loading/Building model...")
    if CHECKPOINT_PATH.exists():
        print("   ✓ Loading existing checkpoint...")
        model = load_model(CHECKPOINT_PATH, compile=False)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("   ✓ Checkpoint loaded! Continuing training...")
    else:
        print("   ✓ No checkpoint found. Building FAST model...")
        model = build_fast_model()
        print("   ✓ Fast model built!")
    
    # Setup callbacks
    print("\n[3/4] Setting up callbacks...")
    callbacks = [
        ModelCheckpoint(
            str(CHECKPOINT_PATH),
            monitor='val_accuracy',
            save_best_only=False,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print(f"\n[4/4] Training for {EPOCHS_PER_SESSION} epoch...")
    print(f"   Session will train epoch {total_epochs_done + 1}")
    print(f"   ⚡ Estimated time: ~10-15 minutes per epoch")
    print("-"*70)
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS_PER_SESSION,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    print("\n[5/5] Saving model and progress...")
    model.save(str(CHECKPOINT_PATH))
    print(f"   ✓ Checkpoint saved: {CHECKPOINT_PATH}")
    
    progress['total_epochs_completed'] += EPOCHS_PER_SESSION
    
    for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
        progress['history'][key].extend(history.history[key])
    
    current_best = max(history.history['val_accuracy'])
    if current_best > progress['best_val_accuracy']:
        progress['best_val_accuracy'] = current_best
        model.save(MODEL_DIR / 'best_model.h5')
        print(f"   🎉 New best model saved! Accuracy: {current_best:.4f}")
    
    save_training_progress(progress)
    plot_training_history(progress)
    
    # Summary
    print("\n" + "="*70)
    print("✅ FAST TRAINING SESSION COMPLETE!")
    print("="*70)
    print(f"📈 This Session:")
    print(f"   • Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"   • Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"\n📊 Overall Progress:")
    print(f"   • Total Epochs: {progress['total_epochs_completed']}")
    print(f"   • Best Accuracy: {progress['best_val_accuracy']:.4f}")
    print(f"\n💾 Files:")
    print(f"   • Checkpoint: {CHECKPOINT_PATH}")
    print(f"   • Best Model: {MODEL_DIR / 'best_model.h5'}")
    print("="*70)
    print("\n🔄 To continue: py -3.11 train_fast.py")
    print("⚡ Each epoch: ~10-15 minutes (8x faster!)")
    print("🎯 Target: 10-15 epochs for good accuracy")
    print("="*70)

if __name__ == "__main__":
    main()
