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

Save format:
  SavedModel directory (NOT .keras).
  The .keras v3 format re-downloads the TF-Hub module at load time and
  fails to restore variables when weights were originally trainable —
  producing "Layer 'keras_layer' expected 1 variables, but received 0".
  SavedModel bundles all hub weights inline and loads reliably on Render.
"""

import os
import time
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Module-level singleton — populated once on startup.
# After tf.saved_model.load() we keep the raw SavedModel object and call
# its __call__ / serving function directly (not model.predict).
_model = None          # raw SavedModel or Keras model object
_infer = None          # callable: text array → raw logit tensor
_model_loaded: bool = False
_load_time_ms: float | None = None


def load_model() -> None:
    """
    Load the SavedModel from disk into the module-level singleton.
    Called once during application lifespan startup.

    We use tf.saved_model.load() (not keras.models.load_model) because the
    model is saved in SavedModel directory format. This avoids the TF-Hub
    variable-restoration bug that occurs with the .keras format.
    """
    global _model, _infer, _model_loaded, _load_time_ms

    logger.info("Loading model from %s …", settings.MODEL_PATH)
    t0 = time.perf_counter()

    try:
        _model = tf.saved_model.load(settings.MODEL_PATH)

        # Resolve the serving callable:
        # tf.saved_model.load gives a trackable with a "serving_default"
        # signature (set by tf.saved_model.save). Fall back to __call__ if
        # the signature key is absent (e.g. older SavedModel exports).
        if hasattr(_model, 'signatures') and 'serving_default' in _model.signatures:
            _sig = _model.signatures['serving_default']
            # signature input key is typically 'keras_tensor' or 'inputs'
            _input_key = list(_sig.structured_input_signature[1].keys())[0]
            def _infer_fn(texts: np.ndarray) -> np.ndarray:
                tensor = tf.constant(texts, dtype=tf.string)
                out = _sig(**{_input_key: tensor})
                # output key is typically 'output_0' or 'dense_1'
                logits = list(out.values())[0]
                return logits.numpy()
            _infer = _infer_fn
            logger.info("Using SavedModel serving_default signature (input key: '%s').", _input_key)
        else:
            # Fallback: direct __call__
            def _infer_fn(texts: np.ndarray) -> np.ndarray:
                tensor = tf.constant(texts, dtype=tf.string)
                return _model(tensor, training=False).numpy()
            _infer = _infer_fn
            logger.info("Using SavedModel __call__ fallback.")

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
        _infer(dummy)
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
    if not _model_loaded or _infer is None:
        raise RuntimeError("Model is not loaded. Cannot run inference.")

    t0 = time.perf_counter()

    # Model expects a numpy array of strings
    arr = np.array([text])
    raw_logit = _infer(arr)          # shape (1, 1) or (1,)

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
