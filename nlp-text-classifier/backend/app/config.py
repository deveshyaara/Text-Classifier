"""
config.py — Application configuration loaded from environment variables.
"""

import os
from typing import List


class Settings:
    # API
    APP_NAME: str = "NLP Text Classifier"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Binary sentiment classifier using TensorFlow + TensorFlow Hub. "
        "Trained on IMDb movie reviews."
    )

    # Model path — relative to this file's directory
    MODEL_PATH: str = os.environ.get(
        "MODEL_PATH",
        # SavedModel directory — NOT .keras file (see model.py for why)
        os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_model"),
    )

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
