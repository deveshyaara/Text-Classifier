"""
train_vit.py — Train a Vision Transformer on CIFAR-10.

Architecture matches Vision_Transformer.ipynb:
  Input       : 32x32x3
  Augmentation: Normalize → Resize(72) → RandomFlip → RandomRotation(0.02) → RandomZoom(0.2)
  Patch size  : 6x6 → 144 patches
  Projection  : 64-dim
  Transformer : 8 layers, 4 heads, MLP[128,64], dropout=0.1
  MLP head    : [2048, 1024], GELU, dropout=0.5
  Output      : 10 logits (CIFAR-10)
  Optimizer   : AdamW lr=0.001, wd=0.0001
  Loss        : SparseCategoricalCrossentropy(from_logits=True)

Training is stopped early if val_accuracy stops improving (patience=10).
Best weights are restored and saved to ../backend/models/vit_model.keras.
Training history is saved to training_history_vit.json for frontend charts.
"""

import json
import os

import keras
import numpy as np
import tensorflow as tf
from keras import layers

# ── Config ────────────────────────────────────────────────────────────────────

NUM_CLASSES    = 10
INPUT_SHAPE    = (32, 32, 3)
IMAGE_SIZE     = 72
PATCH_SIZE     = 6
NUM_PATCHES    = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 144
PROJECTION_DIM = 64
NUM_HEADS      = 4
TRANSFORMER_UNITS   = [128, 64]
TRANSFORMER_LAYERS  = 8
MLP_HEAD_UNITS = [2048, 1024]

LEARNING_RATE = 0.001
WEIGHT_DECAY  = 0.0001
BATCH_SIZE    = 256
NUM_EPOCHS    = 50          # upper bound; early stopping exits before this
PATIENCE      = 10          # stop if val_accuracy doesn't improve for 10 epochs
VAL_SPLIT     = 0.1

MODEL_OUT = "../backend/models/vit_model.keras"
HISTORY_OUT = "training_history_vit.json"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ── Data augmentation (defined here so Normalization can be .adapt()-ed) ──────

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

# ── Custom layers (must use same package tag as the backend adapter) ───────────

@keras.saving.register_keras_serializable(package="vit")
class Patches(layers.Layer):
    """Split an image into non-overlapping (patch_size × patch_size) patches."""

    def __init__(self, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


@keras.saving.register_keras_serializable(package="vit")
class PatchEncoder(layers.Layer):
    """Linearly project patches + add learnable positional embeddings."""

    def __init__(self, num_patches: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return cfg


# ── Model builder ─────────────────────────────────────────────────────────────

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit_classifier():
    inputs = layers.Input(shape=INPUT_SHAPE)
    augmented = data_augmentation(inputs)
    patches = Patches(PATCH_SIZE)(augmented)
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attn = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attn, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    logits = layers.Dense(NUM_CLASSES)(features)

    return keras.Model(inputs=inputs, outputs=logits)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load CIFAR-10
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    print(f"  Train: {x_train.shape}  Test: {x_test.shape}")

    # 2. Fit the Normalization layer
    print("Fitting normalization layer...")
    data_augmentation.layers[0].adapt(x_train)

    # 3. Build & compile
    print("Building Vision Transformer model...")
    model = create_vit_classifier()
    model.summary()

    optimizer = keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    # 4. Callbacks
    os.makedirs("../backend/models", exist_ok=True)
    checkpoint_path = "../backend/models/vit_model.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6,
        ),
        keras.callbacks.CSVLogger("training_log_vit.csv"),
    ]

    # 5. Train
    print(f"\nStarting training (up to {NUM_EPOCHS} epochs, early stopping patience={PATIENCE})...")
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
    )

    # 6. Evaluate on test set
    print("\nEvaluating on test set...")
    results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)
    test_loss     = results[0]
    test_accuracy = results[1]
    test_top5     = results[2]
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy*100:.2f}%")
    print(f"  Top-5 Accuracy: {test_top5*100:.2f}%")

    # 7. Save training history for frontend charts
    hist_data = {
        "epochs":        list(range(1, len(history.history["accuracy"]) + 1)),
        "train_loss":    [float(v) for v in history.history["loss"]],
        "train_accuracy":[float(v) for v in history.history["accuracy"]],
        "val_loss":      [float(v) for v in history.history["val_loss"]],
        "val_accuracy":  [float(v) for v in history.history["val_accuracy"]],
        "test_loss":     float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_top5_accuracy": float(test_top5),
    }
    with open(HISTORY_OUT, "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"\nTraining history saved to {HISTORY_OUT}")
    print(f"Best model saved to      {checkpoint_path}")
    print("\nDone!")
