"""
Traffic Sign Recognition - Model Training Script
=================================================
Train a Convolutional Neural Network (CNN) to recognize traffic signs.

Usage:
    python train_model.py --dataset dataset/ --epochs 30

Author: Traffic Sign Recognition Project
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config


def build_cnn_model(num_classes: int, img_size: int) -> keras.Model:
    """
    Build a CNN model for traffic sign classification.

    Architecture:
        - 3 Convolutional blocks with BatchNorm + MaxPooling + Dropout
        - Global Average Pooling
        - Dense layers with Dropout regularization
        - Softmax output

    Args:
        num_classes: Number of traffic sign classes
        img_size: Input image size (square)

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name='input_layer')

    # --- Block 1 ---
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.Activation('relu', name='act1_1')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.Activation('relu', name='act1_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.25, name='drop1')(x)

    # --- Block 2 ---
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.Activation('relu', name='act2_1')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.Activation('relu', name='act2_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.25, name='drop2')(x)

    # --- Block 3 ---
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.Activation('relu', name='act3_1')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    x = layers.Activation('relu', name='act3_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.3, name='drop3')(x)

    # --- Classifier Head ---
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense')(x)
    x = layers.Activation('relu', name='act_dense')(x)
    x = layers.Dropout(0.5, name='drop_dense')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='TrafficSignCNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_data_generators(dataset_path: str, img_size: int, batch_size: int):
    """
    Create train/validation data generators with augmentation.

    Args:
        dataset_path: Path to dataset directory
        img_size: Target image size
        batch_size: Batch size for training

    Returns:
        Tuple of (train_generator, val_generator, class_names)
    """
    print("\n📂 Setting up data generators...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=Config.VALIDATION_SPLIT
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=Config.VALIDATION_SPLIT
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    class_names = list(train_generator.class_indices.keys())
    print(f"✅ Found {train_generator.samples} training images")
    print(f"✅ Found {val_generator.samples} validation images")
    print(f"✅ Classes detected: {len(class_names)}")

    return train_generator, val_generator, class_names


def get_callbacks(model_save_path: str) -> list:
    """
    Create training callbacks.

    Args:
        model_save_path: Path to save best model

    Returns:
        List of Keras callbacks
    """
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def plot_training_history(history: dict, save_dir: str) -> None:
    """
    Plot and save training accuracy and loss curves.

    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Training History', fontsize=16, fontweight='bold')

    # Accuracy
    axes[0].plot(history['accuracy'], label='Train Accuracy', color='#2196F3', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Val Accuracy', color='#4CAF50', linewidth=2)
    axes[0].set_title('Accuracy Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history['loss'], label='Train Loss', color='#F44336', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', color='#FF9800', linewidth=2)
    axes[1].set_title('Loss Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Training plot saved: {plot_path}")


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_dir: str) -> None:
    """
    Plot and save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_dir: Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5
    )
    ax.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved: {plot_path}")


class _NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy scalar types (e.g. float32, int64)."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_model_metadata(
    model_path: str,
    class_names: list,
    history: dict,
    val_accuracy: float,
    img_size: int
) -> None:
    """Save model metadata as JSON for use by the Flask app."""
    metadata = {
        'model_path': model_path,
        'class_names': class_names,
        'img_size': img_size,
        'num_classes': len(class_names),
        'val_accuracy': round(float(val_accuracy), 4),
        'trained_at': datetime.now().isoformat(),
        'epochs_trained': len(history['accuracy']),
        'final_train_accuracy': round(float(history['accuracy'][-1]), 4),
        'final_val_accuracy': round(float(history['val_accuracy'][-1]), 4)
    }

    metadata_path = os.path.join(os.path.dirname(model_path), 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, cls=_NumpyEncoder)

    print(f"📋 Model metadata saved: {metadata_path}")

    # Save training history
    history_path = os.path.join(os.path.dirname(model_path), 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, cls=_NumpyEncoder)
    print(f"📋 Training history saved: {history_path}")


def train(dataset_path: str, model_save_path: str, epochs: int, img_size: int, batch_size: int):
    """
    Main training function.

    Args:
        dataset_path: Path to dataset directory
        model_save_path: Path to save trained model
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size
    """
    print("=" * 60)
    print("  Traffic Sign Recognition - Model Training")
    print("=" * 60)
    print(f"  Dataset:    {dataset_path}")
    print(f"  Model path: {model_save_path}")
    print(f"  Epochs:     {epochs}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Batch size: {batch_size}")
    print("=" * 60)

    # Validate dataset path
    if not os.path.exists(dataset_path):
        print(f"\n❌ ERROR: Dataset path not found: {dataset_path}")
        print("Please ensure your dataset is placed in the 'dataset/' directory.")
        print("Expected structure:")
        print("  dataset/")
        print("    bicycle/")
        print("    no_entry/")
        print("    ... (one folder per class)")
        sys.exit(1)

    # Check for class folders
    class_dirs = [d for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d))]
    if not class_dirs:
        print(f"\n❌ ERROR: No class folders found in {dataset_path}")
        sys.exit(1)

    print(f"\n✅ Found {len(class_dirs)} classes in dataset")

    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Create data generators
    train_gen, val_gen, class_names = create_data_generators(
        dataset_path, img_size, batch_size
    )

    # Build model
    print(f"\n🏗️  Building CNN model...")
    model = build_cnn_model(num_classes=len(class_names), img_size=img_size)
    model.summary()

    # Callbacks
    callbacks = get_callbacks(model_save_path)

    # Train
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("-" * 60)

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n📈 Evaluating model on validation set...")
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
    print(f"✅ Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"✅ Validation Loss:     {val_loss:.4f}")

    # Generate predictions for confusion matrix
    print("\n🔍 Generating predictions for analysis...")
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes[:len(y_pred)]

    # Save plots
    model_dir = os.path.dirname(model_save_path)
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'img')

    plot_training_history(history.history, static_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, static_dir)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n📊 Classification Report:")
    print(report)

    report_path = os.path.join(model_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Traffic Sign Recognition - Classification Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Validation Accuracy: {val_accuracy * 100:.2f}%\n\n")
        f.write(report)
    print(f"📋 Report saved: {report_path}")

    # Save metadata
    save_model_metadata(
        model_save_path,
        class_names,
        history.history,
        val_accuracy,
        img_size
    )

    print("\n" + "=" * 60)
    print(f"  ✅ Training Complete!")
    print(f"  📦 Model saved to: {model_save_path}")
    print(f"  🎯 Final Accuracy:  {val_accuracy * 100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Traffic Sign Recognition CNN'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=Config.DATASET_PATH,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=Config.MODEL_PATH,
        help='Path to save trained model (.h5)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=Config.EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=Config.IMG_SIZE,
        help='Image size (square)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=Config.BATCH_SIZE,
        help='Training batch size'
    )

    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        model_save_path=args.model_path,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
