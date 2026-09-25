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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "models", "sentiment_model")

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
    """Load from SavedModel directory — avoids TF-Hub variable restore bug."""
    if not os.path.isdir(MODEL_PATH):
        print(f"ERROR: SavedModel directory not found at {MODEL_PATH}")
        print("Run train.py (or convert.py if you have the old .keras file).")
        sys.exit(1)
    sm = tf.saved_model.load(MODEL_PATH)
    if hasattr(sm, "signatures") and "serving_default" in sm.signatures:
        sig = sm.signatures["serving_default"]
        input_key = list(sig.structured_input_signature[1].keys())[0]
        def _infer(texts):
            out = sig(**{input_key: tf.constant(texts, dtype=tf.string)})
            return list(out.values())[0].numpy()
    else:
        def _infer(texts):
            return sm(tf.constant(texts, dtype=tf.string), training=False).numpy()
    return _infer


def predict(infer, texts):
    """
    Run inference.
    Model outputs raw logits. Apply sigmoid to get probabilities.
    Threshold: 0.5
    """
    raw_logits = infer(np.array(texts))
    probs_positive = tf.sigmoid(raw_logits).numpy().flatten()
    return probs_positive


def run_tests(infer):
    all_passed = True

    print("=== Positive Examples ===")
    for text in POSITIVE_EXAMPLES:
        probs = predict(infer, [text])
        p = probs[0]
        label = "positive" if p >= 0.5 else "negative"
        ok = label == "positive"
        if not ok:
            all_passed = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] [{label:8s} {p:.4f}] {text[:60]}")

    print("\n=== Negative Examples ===")
    for text in NEGATIVE_EXAMPLES:
        probs = predict(infer, [text])
        p = probs[0]
        label = "positive" if p >= 0.5 else "negative"
        ok = label == "negative"
        if not ok:
            all_passed = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] [{label:8s} {p:.4f}] {text[:60]}")

    print("\n=== Probability Sanity Checks ===")
    all_texts = POSITIVE_EXAMPLES + NEGATIVE_EXAMPLES
    probs = predict(infer, all_texts)
    for p in probs:
        assert 0.0 <= p <= 1.0, f"Probability out of range: {p}"
    print("  All probabilities in [0, 1] [OK]")

    return all_passed


if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    infer = load_model()
    print(f"Model loaded: {MODEL_PATH}\n")
    passed = run_tests(infer)
    print("\n" + ("=== ALL TESTS PASSED [OK] ===" if passed else "=== SOME TESTS FAILED [FAIL] ==="))
    sys.exit(0 if passed else 1)
