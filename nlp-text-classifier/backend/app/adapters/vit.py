"""
adapters/vit.py — Vision Transformer (CIFAR-10) adapter.

Model details (from Vision_Transformer.ipynb):
  Dataset        : CIFAR-10 (tf.keras.datasets.cifar10)
  Input shape    : 32×32×3
  Resize to      : 72×72
  Patch size     : 6×6  →  144 patches
  Projection dim : 64
  Transformer layers : 8
  Attention heads    : 4
  Transformer units  : [128, 64]
  MLP head units     : [2048, 1024]
  Dropout (attn/MLP) : 0.1 / 0.5
  Loss           : SparseCategoricalCrossentropy(from_logits=True)
  Test accuracy  : 81.78%
  Top-5 accuracy : 99.06%
  Classes        : airplane, automobile, bird, cat, deer,
                   dog, frog, horse, ship, truck
"""

import io
import time
import httpx
from typing import Any, Dict, List, Optional

from app.adapters.base import BaseModelAdapter
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# ── CIFAR-10 labels ───────────────────────────────────────────────────────────

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

INPUT_SIZE = 32   # Native CIFAR-10 resolution fed into the model

# ── Optional heavy deps ───────────────────────────────────────────────────────

try:
    import keras
    import numpy as np
    import tensorflow as tf
    from PIL import Image as PILImage
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

# ── Custom Keras layers (must be registered BEFORE load_model is called) ──────
#
# These replicate the exact layer definitions used during training.
# The @register_keras_serializable decorator lets Keras resolve the class
# by name when deserialising the .keras archive.

if _DEPS_AVAILABLE:

    @keras.saving.register_keras_serializable(package="vit")
    class Patches(tf.keras.layers.Layer):
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

        def get_config(self) -> Dict[str, Any]:
            config = super().get_config()
            config.update({"patch_size": self.patch_size})
            return config

    @keras.saving.register_keras_serializable(package="vit")
    class PatchEncoder(tf.keras.layers.Layer):
        """Linearly project patches + add learnable position embeddings."""

        def __init__(self, num_patches: int, projection_dim: int, **kwargs):
            super().__init__(**kwargs)
            self.num_patches = num_patches
            self.projection_dim = projection_dim
            self.projection = tf.keras.layers.Dense(units=projection_dim)
            self.position_embedding = tf.keras.layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            return self.projection(patch) + self.position_embedding(positions)

        def get_config(self) -> Dict[str, Any]:
            config = super().get_config()
            config.update({
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim,
            })
            return config


# ── Adapter ───────────────────────────────────────────────────────────────────

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
        if settings.HF_SPACE_URL:
            logger.info("ViT model configured to use remote Hugging Face Space: %s", settings.HF_SPACE_URL)
            self._model = "hf_space_proxy"
            return
            
        if not settings.ENABLE_VIT:
            logger.warning("ViT model loading disabled via ENABLE_VIT=false (useful for low-RAM environments like Render Free Tier).")
            return

        path = settings.VIT_MODEL_PATH
        logger.info("Loading ViT model from %s …", path)
        if not _DEPS_AVAILABLE:
            raise RuntimeError("TensorFlow / Pillow not installed.")
        # Patches and PatchEncoder are registered above; Keras can resolve them.
        self._model = tf.keras.models.load_model(path)
        # Warm-up pass
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
        payload: bytes — raw image file bytes (PNG / JPG / WEBP / GIF)
        Returns a dict compatible with ViTPredictResponse.
        """
        if not self.is_loaded():
            raise RuntimeError("ViT model is not loaded.")
            
        # ── 1. Proxy to Hugging Face Space ────────────────────────────────────
        if self._model == "hf_space_proxy":
            t0 = time.perf_counter()
            try:
                # The payload is bytes, so we can send it directly as a file upload
                files = {'file': ('image.jpg', payload, 'image/jpeg')}
                response = httpx.post(f"{settings.HF_SPACE_URL}/predict", files=files, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
                # Add Render tracking metrics
                data["model_id"] = self.model_id
                # Replace the HF inference time with the total round-trip time
                data["inference_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                return data
            except Exception as exc:
                raise RuntimeError(f"Hugging Face Space inference failed: {exc}") from exc

        # ── 2. Local Inference ────────────────────────────────────────────────
        # Decode image → (1, 32, 32, 3) float32 numpy array
        try:
            img = PILImage.open(io.BytesIO(payload)).convert("RGB")
            img = img.resize((INPUT_SIZE, INPUT_SIZE), PILImage.LANCZOS)
            arr = np.array(img, dtype=np.float32)   # (32, 32, 3)
            batch = np.expand_dims(arr, axis=0)      # (1, 32, 32, 3)
        except Exception as exc:
            raise ValueError(f"Invalid image: {exc}") from exc

        t0 = time.perf_counter()
        try:
            logits = self._model(batch, training=False).numpy()[0]  # (10,)
        except Exception as exc:
            raise RuntimeError(f"ViT inference failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Numerically stable softmax
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
            "framework": "TensorFlow 2 / Keras",
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
                    {"name": "Transformer Encoder ×8", "detail": "LayerNorm → MultiHeadAttention(heads=4, key_dim=64, dropout=0.1) → Add → LayerNorm → MLP[128,64] → Add"},
                    {"name": "LayerNorm + Flatten + Dropout(0.5)", "detail": "Representation layer"},
                    {"name": "MLP Head", "detail": "[2048, 1024] with GELU + Dropout(0.5)"},
                    {"name": "Dense(10)", "detail": "Raw logit output — 10 CIFAR-10 classes"},
                ],
            },
            "metrics": {
                "test_accuracy": 0.8178,
                "top5_accuracy": 0.9906,
            },
            "training": {
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "loss": "SparseCategoricalCrossentropy(from_logits=True)",
                "epochs": 50,
                "batch_size": 256,
                "val_split": 0.1,
            },
        }
