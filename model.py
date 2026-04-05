"""
model.py - CNN Model Creation and Training
==========================================
This file builds and trains a Convolutional Neural Network (CNN) to classify
facial expressions as "Stressed" or "Not Stressed".

Dataset: FER-2013 (Facial Expression Recognition)
- Stressed  → maps to: angry, disgust, fear (FER labels: 0, 1, 2)
- Not Stressed → maps to: happy, neutral, surprise (FER labels: 3, 6, 5)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE   = 48        # FER-2013 images are 48×48 pixels
BATCH_SIZE = 64
EPOCHS     = 30
NUM_CLASSES = 2        # Stressed / Not Stressed
MODEL_PATH  = "stress_model.h5"

# FER-2013 label mapping → binary stress label
# 0=Angry, 1=Disgust, 2=Fear  → Stressed   (label 1)
# 3=Happy, 4=Sad, 5=Surprise, 6=Neutral → Not Stressed (label 0)
# Note: We treat Sad separately; you can tweak this mapping as needed.
STRESSED_LABELS     = {0, 1, 2}     # angry, disgust, fear
NOT_STRESSED_LABELS = {3, 4, 5, 6}  # happy, sad, surprise, neutral


# ─────────────────────────────────────────────
# 2. LOAD & PREPROCESS FER-2013 DATASET
# ─────────────────────────────────────────────
def load_fer2013(csv_path: str):
    """
    Load the FER-2013 CSV file and return images + binary stress labels.

    FER-2013 CSV format:
        emotion, pixels, Usage
        0, "70 80 82 ...", Training

    Args:
        csv_path: path to 'fer2013.csv'

    Returns:
        X : np.ndarray of shape (N, 48, 48, 1), dtype float32, values 0–1
        y : np.ndarray of shape (N,), dtype int32  (0 = not stressed, 1 = stressed)
    """
    import pandas as pd

    print(f"[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    images, labels = [], []

    for _, row in df.iterrows():
        fer_label = int(row["emotion"])

        # Skip emotions we don't map clearly (optional: keep all)
        if fer_label not in STRESSED_LABELS and fer_label not in NOT_STRESSED_LABELS:
            continue

        # Convert space-separated pixel string → 48×48 array
        pixels = np.array(row["pixels"].split(), dtype=np.float32)
        image  = pixels.reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0  # normalise to [0,1]

        # Binary label
        stress_label = 1 if fer_label in STRESSED_LABELS else 0

        images.append(image)
        labels.append(stress_label)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    print(f"[INFO] Dataset loaded — {len(X)} samples")
    print(f"[INFO] Stressed: {np.sum(y==1)}, Not Stressed: {np.sum(y==0)}")
    return X, y


# ─────────────────────────────────────────────
# 3. BUILD CNN MODEL
# ─────────────────────────────────────────────
def build_model() -> tf.keras.Model:
    """
    Build a simple CNN suitable for 48×48 grayscale facial images.
    Architecture: 3 Conv blocks → Flatten → Dense → Output
    """
    model = Sequential([
        # ── Block 1 ──
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Block 2 ──
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Block 3 ──
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        # ── Classifier ──
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),  # [not_stressed, stressed]
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


# ─────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────
def train(csv_path: str):
    """Full training pipeline."""
    # 4a. Load data
    X, y = load_fer2013(csv_path)

    # 4b. Train / validation split (80 / 20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}, Validation: {len(X_val)}")

    # 4c. Data augmentation (helps the model generalise)
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    datagen.fit(X_train)

    # 4d. Build model
    model = build_model()

    # 4e. Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    # 4f. Fit
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print(f"\n[INFO] Model saved to: {MODEL_PATH}")
    plot_history(history)
    return model, history


# ─────────────────────────────────────────────
# 5. PLOT TRAINING HISTORY
# ─────────────────────────────────────────────
def plot_history(history):
    """Save accuracy & loss curves as training_history.png."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train Acc")
    ax1.plot(history.history["val_accuracy"], label="Val Acc")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"],     label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("[INFO] Training curves saved to training_history.png")


# ─────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model.py <path_to_fer2013.csv>")
        print("Example: python model.py data/fer2013.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    train(csv_path)
