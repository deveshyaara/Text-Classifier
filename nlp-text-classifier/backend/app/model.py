"""
model.py — Production inference module.

Design rules:
  - Model is loaded ONCE at application startup (via lifespan context).
  - All requests share the same in-memory model instance.
  - Model output is a raw logit → sigmoid() → probability.
  - Threshold = 0.5  (label 1 = positive, label 0 = negative).

Architecture (verified from notebook):
  Input text (string)
       ↓
  TF Hub Swivel 20-dim embedding  (trainable=True in training)
       ↓
  Dense(16, activation='relu')
       ↓
  Dense(1)   ← raw logit, no activation
       ↓
  sigmoid(logit) → probability of class=1 (positive)
"""

import os
import time
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Module-level singleton — populated once on startup
_model: tf.keras.Model | None = None
_model_loaded: bool = False
_load_time_ms: float | None = None


def load_model() -> None:
    """
    Load the Keras model from disk into the module-level singleton.
    Called once during application lifespan startup.
    """
    global _model, _model_loaded, _load_time_ms

    logger.info("Loading model from %s …", settings.MODEL_PATH)
    t0 = time.perf_counter()

    try:
        import tensorflow_hub as hub
        _model = tf.keras.models.load_model(
            settings.MODEL_PATH,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
        elapsed = (time.perf_counter() - t0) * 1000
        _load_time_ms = elapsed
        _model_loaded = True
        logger.info("Model loaded in %.1f ms", elapsed)

        # Warm-up inference — avoids cold-start latency on first real request
        _warmup()
    except Exception as exc:
        _model_loaded = False
        logger.error("Failed to load model: %s", exc)
        raise


def _warmup() -> None:
    """Run a single inference to warm up TF graph compilation."""
    try:
        dummy = np.array(["warm up"])
        _model.predict(dummy, verbose=0)
        logger.info("Model warm-up complete.")
    except Exception as exc:
        logger.warning("Model warm-up failed (non-fatal): %s", exc)


def is_loaded() -> bool:
    """Return True if the model singleton is ready."""
    return _model_loaded


def predict(text: str) -> dict:
    """
    Run inference on a single preprocessed text string.

    Parameters
    ----------
    text : str
        Preprocessed input (from preprocessing.preprocess()).

    Returns
    -------
    dict with keys:
        prediction      str   "positive" | "negative"
        confidence      float probability of the predicted class (0–1)
        probabilities   dict  {"positive": float, "negative": float}
        inference_time_ms float  actual measured latency
    """
    if not _model_loaded or _model is None:
        raise RuntimeError("Model is not loaded. Cannot run inference.")

    t0 = time.perf_counter()

    # Model expects a numpy array of strings
    arr = np.array([text])
    raw_logit = _model.predict(arr, verbose=0)          # shape (1, 1)

    # Convert logit → probability (model uses from_logits=True during training)
    prob_positive = float(tf.sigmoid(raw_logit[0][0]).numpy())
    prob_negative = 1.0 - prob_positive

    predicted_class = 1 if prob_positive >= settings.CLASSIFICATION_THRESHOLD else 0
    prediction_label = settings.LABEL_MAP[predicted_class]
    confidence = prob_positive if predicted_class == 1 else prob_negative

    inference_time_ms = (time.perf_counter() - t0) * 1000

    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 6),
        "probabilities": {
            "positive": round(prob_positive, 6),
            "negative": round(prob_negative, 6),
        },
        "inference_time_ms": round(inference_time_ms, 2),
    }
