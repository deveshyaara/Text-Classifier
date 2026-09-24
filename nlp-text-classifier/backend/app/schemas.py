"""
schemas.py — Pydantic request/response models.
"""

from pydantic import BaseModel, field_validator
from app.config import settings


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace-only.")
        if len(v.strip()) > settings.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text exceeds maximum length of {settings.MAX_TEXT_LENGTH} characters."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "This movie was absolutely brilliant. The acting was fantastic."}
            ]
        }
    }


class Probabilities(BaseModel):
    positive: float
    negative: float


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Probabilities
    inference_time_ms: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": "positive",
                    "confidence": 0.964,
                    "probabilities": {"positive": 0.964, "negative": 0.036},
                    "inference_time_ms": 43.2,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class RootResponse(BaseModel):
    name: str
    version: str
    status: str
