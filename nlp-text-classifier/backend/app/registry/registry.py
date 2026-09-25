"""
registry/registry.py — Central model registry.

All model adapters are registered here.  The registry is instantiated once
and shared via the FastAPI app state.

Adding a new model:
  1. Implement BaseModelAdapter in app/adapters/your_model.py
  2. Import it here
  3. Add it to _ADAPTERS list
  4. That's it — routes, metadata API, and frontend registry pick it up automatically.
"""

import os
from typing import Dict, List, Optional

from app.adapters.base import BaseModelAdapter
from app.adapters.nlp import NLPAdapter
from app.adapters.vit import ViTAdapter
from app.logging_config import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Singleton registry that holds all model adapters.

    Models are loaded eagerly at startup.  If a model's file is missing
    its adapter is still registered but reports status='unavailable'.
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, BaseModelAdapter] = {}

        _ADAPTERS: List[BaseModelAdapter] = [
            NLPAdapter(),
            ViTAdapter(),
        ]

        for adapter in _ADAPTERS:
            self._adapters[adapter.model_id] = adapter

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Attempt to load every registered adapter.  Errors are logged but do not crash."""
        for model_id, adapter in self._adapters.items():
            try:
                adapter.load()
            except FileNotFoundError:
                logger.warning(
                    "Model '%s': weights file not found — marking unavailable.", model_id
                )
            except Exception as exc:
                logger.error("Model '%s' failed to load: %s", model_id, exc)

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, model_id: str) -> Optional[BaseModelAdapter]:
        return self._adapters.get(model_id)

    def all(self) -> List[BaseModelAdapter]:
        return list(self._adapters.values())

    def any_loaded(self) -> bool:
        return any(a.is_loaded() for a in self._adapters.values())

    def all_statuses(self) -> Dict[str, bool]:
        return {mid: a.is_loaded() for mid, a in self._adapters.items()}
