"""
conftest.py — pytest configuration.

Patches out tensorflow so tests run without TF installed.
The model is mocked in individual test files.
"""

import sys
from unittest.mock import MagicMock

# Mock tensorflow and tensorflow_hub before any app imports
# This allows tests to run in CI without TF installed
tf_mock = MagicMock()
tf_mock.keras.models.load_model = MagicMock()
tf_mock.sigmoid = MagicMock(return_value=MagicMock(numpy=lambda: 0.96))

sys.modules.setdefault("tensorflow", tf_mock)
sys.modules.setdefault("tensorflow.keras", tf_mock.keras)
sys.modules.setdefault("tensorflow_hub", MagicMock())
sys.modules.setdefault("numpy", __import__("numpy"))
