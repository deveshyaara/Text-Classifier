"""
config.py — Application configuration loaded from environment variables.
"""

import os
from typing import List


class Settings:
    # API
    APP_NAME: str = "AI Model Lab API"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = (
        "Unified inference backend for the AI Model Lab. "
        "Supports multiple models including NLP Sentiment and Vision Transformer."
    )

    # Model paths — relative to this file's directory
    MODEL_PATH: str = os.environ.get(
        "MODEL_PATH",
        # SavedModel directory — NOT .keras file
        os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_model"),
    )
    
    VIT_MODEL_PATH: str = os.environ.get(
        "VIT_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "models", "vit_model.keras"),
    )
    
    # Hugging Face Space URL (e.g. "https://your-username-space-name.hf.space")
    HF_SPACE_URL: str = os.environ.get("HF_SPACE_URL", "").rstrip("/")
    
    # Toggle to disable ViT model on memory-constrained environments (like Render Free Tier)
    ENABLE_VIT: bool = os.environ.get("ENABLE_VIT", "true").lower() == "true"

    # CORS
    FRONTEND_URL: str = os.environ.get("FRONTEND_URL", "http://localhost:3000")

    # Input validation
    MAX_TEXT_LENGTH: int = int(os.environ.get("MAX_TEXT_LENGTH", "10000"))
    MIN_TEXT_LENGTH: int = 1

    # Classification threshold (logit → sigmoid(logit) >= THRESHOLD → positive)
    CLASSIFICATION_THRESHOLD: float = 0.5

    # Labels — verified from notebook: 0=negative, 1=positive
    LABEL_MAP = {0: "negative", 1: "positive"}

    @property
    def allowed_origins(self) -> List[str]:
        origins = [self.FRONTEND_URL]
        # Always allow localhost variants during development
        origins += [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
        return list(set(origins))


settings = Settings()
