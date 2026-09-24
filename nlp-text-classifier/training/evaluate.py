"""
evaluate.py — Standalone evaluation script.

Loads the saved model and evaluates it on the IMDb test split.
Prints detailed metrics and saves a report.

Run:
    python evaluate.py
"""

import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "models", "sentiment_model.keras")
BATCH_SIZE = 100


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    print(f"Loading model from {MODEL_PATH} …")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
    return model


def load_test_data():
    print("Loading IMDb test data …")
    _, _, test_data = tfds.load(
        name="imdb_reviews",
        split=("test[:60%]", "test[60%:]", "test"),
        as_supervised=True,
    )
    return test_data


def evaluate(model, test_data):
    print("Evaluating …")
    results = model.evaluate(test_data.batch(BATCH_SIZE), verbose=2)
    metrics = dict(zip(model.metrics_names, results))
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.6f}")

    # Additional: compute predictions to get TP/FP/TN/FN
    all_probs, all_labels = [], []
    for texts, labels in test_data.batch(BATCH_SIZE):
        raw = model.predict(texts, verbose=0)
        probs = tf.sigmoid(raw).numpy().flatten()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    tp = int(np.sum((preds == 1) & (all_labels == 1)))
    fp = int(np.sum((preds == 1) & (all_labels == 0)))
    tn = int(np.sum((preds == 0) & (all_labels == 0)))
    fn = int(np.sum((preds == 0) & (all_labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\n  Confusion matrix:")
    print(f"    TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    report = {
        **metrics,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }

    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    return report


if __name__ == "__main__":
    model = load_model()
    test_data = load_test_data()
    evaluate(model, test_data)
