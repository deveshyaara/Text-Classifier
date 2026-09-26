"""
conftest.py — pytest configuration for the AI Model Lab API.

Mocks out heavy dependencies (TensorFlow, Keras, Pillow) so all tests
run quickly in CI without needing GPU/TF installed.
"""

import sys
from unittest.mock import MagicMock

# ── Mock TensorFlow / Keras / Pillow before any app imports ──────────────────

keras_mock = MagicMock()
tf_mock = MagicMock()
tf_mock.keras = keras_mock

pillow_mock = MagicMock()

sys.modules.setdefault("tensorflow", tf_mock)
sys.modules.setdefault("tensorflow.keras", keras_mock)
sys.modules.setdefault("tensorflow_hub", MagicMock())
sys.modules.setdefault("keras", keras_mock)
sys.modules.setdefault("PIL", pillow_mock)
sys.modules.setdefault("PIL.Image", pillow_mock.Image)
sys.modules.setdefault("numpy", __import__("numpy"))
