"""
adapters/vit.py — Vision Transformer (CIFAR-10) adapter.

Verified model details from Vision_Transformer.ipynb (executable cells only):
  Dataset        : CIFAR-10 (tf.keras.datasets.cifar10)
  Input shape    : 32×32×3
  Resize to      : 72×72
  Patch size     : 6×6  →  144 patches
  Projection dim : 64
  Transformer layers : 8
  Attention heads    : 4
  Transformer units  : [128, 64]
  MLP head units     : [2048, 1024]
  Dropout (attn/MLP) : 0.1
  Dropout (head)     : 0.5
  Augmentation   : Normalization, Resize(72,72), RandomFlip, RandomRotation(0.02), RandomZoom(0.2)
  Optimizer      : AdamW (tensorflow_addons) lr=0.001, wd=0.0001
  Loss           : SparseCategoricalCrossentropy(from_logits=True)
  Epochs         : 50  |  Batch: 256  |  Val split: 10%
  Test accuracy  : 81.78%
  Top-5 accuracy : 99.06%
  Classes        : airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""

import io
import time
from typing import Any, Dict, List, Optional

from app.adapters.base import BaseModelAdapter
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# CIFAR-10 class names — exact order from the dataset
CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Inference image dimensions (model input after resize in augmentation layer)
IMAGE_SIZE = 72   # model resizes internally; we send 32×32 normalised
INPUT_SIZE = 32   # CIFAR-10 native resolution

try:
    import numpy as np
    import tensorflow as tf
    from PIL import Image as PILImage
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


class ViTAdapter(BaseModelAdapter):
    """Adapter for the CIFAR-10 Vision Transformer Keras model."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return "vision-transformer"

    @property
    def display_name(self) -> str:
        return "Vision Transformer"

    @property
    def model_type(self) -> str:
        return "image"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        path = settings.VIT_MODEL_PATH
        logger.info("Loading ViT model from %s …", path)
        if not _DEPS_AVAILABLE:
            raise RuntimeError("TensorFlow / Pillow not installed.")
        # Load as SavedModel directory (saved with tf.saved_model.save or model.save())
        self._model = tf.keras.models.load_model(path)
        # Warm-up pass with a dummy 32×32×3 image
        try:
            dummy = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
            _ = self._model(dummy, training=False)
        except Exception:
            pass
        logger.info("ViT model loaded and warmed up.")

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, payload: Any) -> Dict[str, Any]:
        """
        payload: bytes — raw image file bytes (PNG / JPG / WEBP)
        Returns unified dict compatible with ViTPredictResponse.
        """
        if not self.is_loaded():
            raise RuntimeError("ViT model is not loaded.")

        # Decode image → numpy array (32×32×3, uint8)
        try:
            img = PILImage.open(io.BytesIO(payload)).convert("RGB")
            img = img.resize((INPUT_SIZE, INPUT_SIZE), PILImage.LANCZOS)
            arr = np.array(img, dtype=np.float32)          # (32,32,3)
            batch = np.expand_dims(arr, axis=0)            # (1,32,32,3)
        except Exception as exc:
            raise ValueError(f"Invalid image: {exc}") from exc

        t0 = time.perf_counter()
        try:
            logits = self._model(batch, training=False).numpy()[0]  # (10,)
        except Exception as exc:
            raise RuntimeError(f"ViT inference failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        top_k = 5
        top_indices = np.argsort(probs)[::-1][:top_k].tolist()
        top_predictions = [
            {"class": CIFAR10_CLASSES[i], "probability": float(probs[i])}
            for i in top_indices
        ]

        predicted_idx = int(np.argmax(probs))
        predicted_class = CIFAR10_CLASSES[predicted_idx]
        confidence = float(probs[predicted_idx])

        return {
            "model_id": self.model_id,
            "prediction": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "all_probabilities": {
                cls: float(probs[i]) for i, cls in enumerate(CIFAR10_CLASSES)
            },
            "inference_time_ms": round(elapsed_ms, 2),
        }

    # ── Metadata ──────────────────────────────────────────────────────────────

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": self.model_id,
            "name": self.display_name,
            "type": self.model_type,
            "framework": "TensorFlow 2",
            "status": "live" if self.is_loaded() else "unavailable",
            "dataset": "CIFAR-10",
            "task": "10-class image classification",
            "classes": CIFAR10_CLASSES,
            "architecture": {
                "summary": "Patch Extraction → Patch Encoding → 8× Transformer Encoder → MLP Head → 10 classes",
                "layers": [
                    {"name": "Input", "detail": "32×32×3 RGB image"},
                    {"name": "Data Augmentation", "detail": "Normalize → Resize 72×72 → RandomFlip → RandomRotation(0.02) → RandomZoom(0.2)"},
                    {"name": "Patch Extraction", "detail": "6×6 patches → 144 patches per image"},
                    {"name": "Patch Encoding", "detail": "Linear projection to dim=64 + learnable positional embedding"},
                    {"name": "Transformer Encoder ×8", "detail": "LayerNorm → MultiHeadAttention(heads=4, dim=64, dropout=0.1) → Add → LayerNorm → MLP[128,64] → Add"},
                    {"name": "LayerNorm + Flatten + Dropout(0.5)", "detail": "Representation layer"},
                    {"name": "MLP Head", "detail": "[2048, 1024] with GELU + Dropout(0.5)"},
                    {"name": "Dense(10)", "detail": "Logit output, 10 CIFAR-10 classes"},
                ],
            },
            "metrics": {
                "test_accuracy": 0.8178,
                "top5_accuracy": 0.9906,
            },
            "training": {
                "optimizer": "AdamW (tensorflow_addons)",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "loss": "SparseCategoricalCrossentropy(from_logits=True)",
                "epochs": 50,
                "batch_size": 256,
                "val_split": 0.1,
            },
        }
