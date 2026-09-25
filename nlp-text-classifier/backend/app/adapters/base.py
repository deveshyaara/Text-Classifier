"""
adapters/base.py — Abstract base class for all model adapters.

Every model in the AI Lab must implement this interface so the registry
and routes can treat all models uniformly.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModelAdapter(ABC):
    """Common interface every model adapter must satisfy."""

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique slug used in API paths, e.g. 'nlp-sentiment'."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in the UI."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Input modality: 'text' | 'image'."""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory.  Called once at startup (or lazily)."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True iff the model is ready for inference."""

    # ── Inference ─────────────────────────────────────────────────────────────

    @abstractmethod
    def predict(self, payload: Any) -> Dict[str, Any]:
        """
        Run inference on the preprocessed payload.

        Returns a dict that conforms to the unified PredictResponse schema.
        Raises RuntimeError on inference failure.
        """

    # ── Metadata ──────────────────────────────────────────────────────────────

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Return rich metadata for GET /models/{model_id}.

        Must include at minimum:
          id, name, type, framework, status,
          dataset, task, architecture, metrics
        """
