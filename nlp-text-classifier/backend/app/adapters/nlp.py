"""
adapters/nlp.py — NLP Sentiment Classifier adapter.

Wraps the existing SavedModel-based inference logic (previously in model.py)
into the BaseModelAdapter interface so it can live inside the unified registry.

Verified model details (from executable notebook code):
  Dataset    : IMDb Reviews (tensorflow_datasets)
  Task       : Binary sentiment classification
  Architecture: TF-Hub gnews-swivel-20dim → Dense(16, relu) → Dense(1)
  Parameters : 400,373
  Optimizer  : Adam
  Loss       : BinaryCrossentropy(from_logits=True)
  Epochs     : 25 / Batch: 100
  Test acc   : 94.636%  |  Test loss: 0.2265

Note on notebook inconsistency:
  Some markdown cells describe AG News (4-class).
  The executable code uses imdb_reviews (binary).
  The deployed model follows the executable code.
"""

import time
from typing import Any, Dict

from app.adapters.base import BaseModelAdapter
from app.config import settings
from app.logging_config import get_logger
from app.preprocessing import preprocess

logger = get_logger(__name__)

# Lazy TF import so the module can be imported in test environments where TF
# is mocked via sys.modules in conftest.py.
try:
    import numpy as np
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


class NLPAdapter(BaseModelAdapter):
    """Adapter for the IMDb binary sentiment SavedModel."""

    def __init__(self) -> None:
        self._model = None

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return "nlp-sentiment"

    @property
    def display_name(self) -> str:
        return "NLP Sentiment Classifier"

    @property
    def model_type(self) -> str:
        return "text"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        path = settings.MODEL_PATH
        logger.info("Loading NLP model from %s …", path)
        if not _TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed.")
        self._model = tf.saved_model.load(path)
        # Warm-up: one forward pass so the first real request isn't slow.
        try:
            infer = self._model.signatures.get("serving_default") or self._model.__call__
            _ = infer(tf.constant(["warm-up"]))
        except Exception:
            pass
        logger.info("NLP model loaded and warmed up.")

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, payload: Any) -> Dict[str, Any]:
        """
        payload: str — raw user text (validated upstream)
        Returns unified dict compatible with NLPPredictResponse.
        """
        if not self.is_loaded():
            raise RuntimeError("NLP model is not loaded.")

        text = preprocess(payload)
        t0 = time.perf_counter()

        try:
            infer = self._model.signatures.get("serving_default")
            if infer is not None:
                output = infer(tf.constant([text]))
                logit = list(output.values())[0].numpy()[0][0]
            else:
                logit = float(self._model(tf.constant([text])).numpy()[0][0])
        except Exception as exc:
            raise RuntimeError(f"NLP inference failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000
        prob_pos = float(tf.sigmoid(logit).numpy())
        prob_neg = 1.0 - prob_pos
        is_positive = prob_pos >= settings.CLASSIFICATION_THRESHOLD

        return {
            "model_id": self.model_id,
            "prediction": "positive" if is_positive else "negative",
            "confidence": prob_pos if is_positive else prob_neg,
            "probabilities": {"positive": prob_pos, "negative": prob_neg},
            "inference_time_ms": round(elapsed_ms, 2),
        }

    # ── Metadata ──────────────────────────────────────────────────────────────

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": self.model_id,
            "name": self.display_name,
            "type": self.model_type,
            "framework": "TensorFlow 2 + TF Hub",
            "status": "live" if self.is_loaded() else "unavailable",
            "dataset": "IMDb Movie Reviews",
            "task": "Binary sentiment classification",
            "classes": ["positive", "negative"],
            "architecture": {
                "summary": "TF-Hub gnews-swivel-20dim → Dense(16, relu) → Dense(1)",
                "layers": [
                    {"name": "TF Hub Swivel Embedding", "detail": "20-dim token embedding"},
                    {"name": "Dense", "detail": "16 neurons, ReLU activation"},
                    {"name": "Dense", "detail": "1 neuron, logit output"},
                    {"name": "Sigmoid", "detail": "Applied post-inference for probability"},
                ],
            },
            "metrics": {
                "test_accuracy": 0.94636,
                "test_loss": 0.2265,
                "parameters": 400373,
            },
            "training": {
                "optimizer": "Adam",
                "loss": "BinaryCrossentropy(from_logits=True)",
                "epochs": 25,
                "batch_size": 100,
            },
            "implementation_note": (
                "Some notebook markdown sections describe an AG News 4-class classifier. "
                "The executable code uses imdb_reviews (binary). "
                "This deployed model follows the executable code."
            ),
        }
