"""
convert.py — Convert existing .keras model to SavedModel directory format.

WHY:
  The .keras v3 format re-downloads the TF-Hub module at load time and fails
  to restore trainable hub layer variables, producing:
    "Layer 'keras_layer' expected 1 variables, but received 0 variables"
  SavedModel bundles all weights inline and loads reliably on Render/any server.

Run (once):
    python convert.py

Input:  backend/models/sentiment_model.keras  (old file)
Output: backend/models/sentiment_model/       (new directory)
"""

import os
import sys

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_hub as hub

KERAS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "backend", "models", "sentiment_model.keras"
)
SAVED_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "backend", "models", "sentiment_model"
)


def main():
    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.exists(KERAS_PATH):
        print(f"ERROR: .keras file not found at:\n  {KERAS_PATH}")
        print("Make sure you have the old model file before converting.")
        sys.exit(1)

    if os.path.isdir(SAVED_MODEL_PATH):
        print(f"SavedModel already exists at:\n  {SAVED_MODEL_PATH}")
        print("Delete it first if you want to re-convert.")
        sys.exit(0)

    # ── Load old .keras model ─────────────────────────────────────────────────
    print(f"Loading .keras model from:\n  {KERAS_PATH}")
    print("(This will download the TF-Hub module once — ~2 MB)\n")
    model = tf.keras.models.load_model(
        KERAS_PATH,
        custom_objects={"KerasLayer": hub.KerasLayer},
    )
    model.summary()

    # ── Smoke-test before saving ──────────────────────────────────────────────
    import numpy as np
    dummy = np.array(["quick test"])
    logit = model.predict(dummy, verbose=0)
    import math
    prob = 1.0 / (1.0 + math.exp(-float(logit[0][0])))
    print(f"\nSmoke-test logit={logit[0][0]:.4f}  prob={prob:.4f}  [OK]")

    # ── Save as SavedModel ────────────────────────────────────────────────────
    os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
    tf.saved_model.save(model, SAVED_MODEL_PATH)
    print(f"\nSavedModel written to:\n  {SAVED_MODEL_PATH}/")
    print("\nDone. [OK]")
    print("\nNext steps:")
    print("  1. Push backend/models/sentiment_model/ to your repo (or upload to Render).")
    print("  2. Update Render env var:  MODEL_PATH=./models/sentiment_model")
    print("  3. Redeploy on Render.")


if __name__ == "__main__":
    main()
