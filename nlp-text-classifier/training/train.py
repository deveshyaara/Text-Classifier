"""
train.py — Exact reproduction of the original notebook training pipeline.

Dataset:    IMDb Reviews (tensorflow_datasets)
Split:      train=test[:60%], val=test[60%:], test=test (as in notebook)
Embedding:  https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1
Model:      Hub Layer → Dense(16, relu) → Dense(1)
Loss:       BinaryCrossentropy(from_logits=True)
Optimizer:  Adam
Batch:      100
Epochs:     25

Run:
    python train.py
The model is saved to: ../backend/models/sentiment_model.keras
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# ── Paths ────────────────────────────────────────────────────────────────────
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "models", "sentiment_model.keras")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "training_history.json")

# ── Embedding URL (exactly as in notebook) ───────────────────────────────────
EMBEDDING_URL = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

# ── Hyperparameters (exactly as in notebook) ─────────────────────────────────
BATCH_SIZE = 100
EPOCHS = 25
SHUFFLE_BUFFER = 10000


def load_data():
    """Load IMDb reviews exactly as done in the original notebook."""
    print("Loading IMDb Reviews dataset...")
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=("test[:60%]", "test[60%:]", "test"),
        as_supervised=True,
    )
    print(f"  Train batches (15 000 examples / batch {BATCH_SIZE}): {tf.data.experimental.cardinality(train_data)}")
    print(f"  Validation batches: {tf.data.experimental.cardinality(validation_data)}")
    print(f"  Test batches: {tf.data.experimental.cardinality(test_data)}")
    return train_data, validation_data, test_data


def build_model():
    """Build the exact same architecture as the original notebook."""
    hub_layer = hub.KerasLayer(
        EMBEDDING_URL,
        input_shape=[],
        dtype=tf.string,
        trainable=True,
    )
    model = tf.keras.Sequential([
        hub_layer,
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),          # raw logit — no activation
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train(model, train_data, validation_data):
    """Train for 25 epochs with batch size 100 (identical to notebook)."""
    print("\nTraining...")
    history = model.fit(
        train_data.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=validation_data.batch(BATCH_SIZE),
        verbose=1,
    )
    return history


def evaluate(model, test_data):
    """Evaluate on the test split (same split the notebook uses)."""
    print("\nEvaluating on test set...")
    results = model.evaluate(test_data.batch(BATCH_SIZE), verbose=2)
    for name, value in zip(model.metrics_names, results):
        print(f"  {name}: {value:.6f}")
    return results


def save_model(model):
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    model.save(SAVE_PATH)
    print(f"\nModel saved to: {SAVE_PATH}")


def save_history(history):
    data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(HISTORY_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Training history saved to: {HISTORY_PATH}")


def smoke_test(model):
    """Quick inference test to verify the saved pipeline."""
    samples = [
        "This movie was absolutely brilliant. The acting was fantastic.",
        "The film was boring and completely disappointing.",
    ]
    raw_outputs = model.predict(np.array(samples))
    probs = tf.sigmoid(raw_outputs).numpy().flatten()
    print("\nSmoke test predictions:")
    for text, p in zip(samples, probs):
        label = "positive" if p >= 0.5 else "negative"
        print(f"  [{label:8s} {p:.4f}] {text[:60]}")


if __name__ == "__main__":
    train_data, validation_data, test_data = load_data()
    model = build_model()
    history = train(model, train_data, validation_data)
    evaluate(model, test_data)
    save_model(model)
    save_history(history)
    smoke_test(model)
    print("\nDone. ✓")
