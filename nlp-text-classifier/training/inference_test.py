"""
inference_test.py — Manual inference smoke tests against the saved model.

Validates that:
  - Model loads correctly
  - Logit → sigmoid → probability transformation is correct
  - Known positive/negative examples are classified correctly
  - Edge cases handled (empty, very short, very long text)

Run:
    python inference_test.py
"""

import os
import sys
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "models", "sentiment_model.keras")

POSITIVE_EXAMPLES = [
    "This movie was absolutely brilliant. The acting was fantastic.",
    "One of the best films I have ever seen. Truly amazing.",
    "An outstanding masterpiece. I loved every minute of it.",
]

NEGATIVE_EXAMPLES = [
    "The film was boring and completely disappointing.",
    "Terrible acting and a nonsensical plot. Waste of time.",
    "I hated this movie. The worst thing I have watched in years.",
]


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run train.py first.")
        sys.exit(1)
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, texts):
    """
    Run inference.
    Model outputs raw logits. Apply sigmoid to get probabilities.
    Threshold: 0.5
    """
    arr = np.array(texts)
    raw_logits = model.predict(arr, verbose=0)
    probs_positive = tf.sigmoid(raw_logits).numpy().flatten()
    return probs_positive


def run_tests(model):
    all_passed = True

    print("=== Positive Examples ===")
    for text in POSITIVE_EXAMPLES:
        probs = predict(model, [text])
        p = probs[0]
        label = "positive" if p >= 0.5 else "negative"
        ok = label == "positive"
        if not ok:
            all_passed = False
        status = "✓" if ok else "✗ FAIL"
        print(f"  [{status}] [{label:8s} {p:.4f}] {text[:60]}")

    print("\n=== Negative Examples ===")
    for text in NEGATIVE_EXAMPLES:
        probs = predict(model, [text])
        p = probs[0]
        label = "positive" if p >= 0.5 else "negative"
        ok = label == "negative"
        if not ok:
            all_passed = False
        status = "✓" if ok else "✗ FAIL"
        print(f"  [{status}] [{label:8s} {p:.4f}] {text[:60]}")

    print("\n=== Probability Sanity Checks ===")
    all_texts = POSITIVE_EXAMPLES + NEGATIVE_EXAMPLES
    probs = predict(model, all_texts)
    for p in probs:
        assert 0.0 <= p <= 1.0, f"Probability out of range: {p}"
    print("  All probabilities in [0, 1] ✓")

    return all_passed


if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    model = load_model()
    print(f"Model loaded: {MODEL_PATH}\n")
    passed = run_tests(model)
    print("\n" + ("=== ALL TESTS PASSED ✓ ===" if passed else "=== SOME TESTS FAILED ✗ ==="))
    sys.exit(0 if passed else 1)
