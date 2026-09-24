"""
test_predict.py — Tests for POST /predict with a mocked model.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


MOCK_RESULT = {
    "prediction": "positive",
    "confidence": 0.964123,
    "probabilities": {"positive": 0.964123, "negative": 0.035877},
    "inference_time_ms": 43.2,
}


@pytest.fixture
def client():
    with patch("app.model.load_model"):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_with_model():
    """Client where model appears loaded and predict is mocked."""
    with patch("app.model.load_model"):
        from app.main import app
        with patch("app.model._model_loaded", True), \
             patch("app.model.predict", return_value=MOCK_RESULT):
            with TestClient(app) as c:
                yield c


def test_predict_positive_review(client_with_model):
    resp = client_with_model.post(
        "/predict",
        json={"text": "This movie was absolutely brilliant. The acting was fantastic."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "positive" in data["probabilities"]
    assert "negative" in data["probabilities"]
    assert "inference_time_ms" in data
    assert data["prediction"] in ("positive", "negative")
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probabilities"]["positive"] <= 1.0
    assert 0.0 <= data["probabilities"]["negative"] <= 1.0


def test_predict_probabilities_sum_to_one(client_with_model):
    resp = client_with_model.post(
        "/predict",
        json={"text": "A fantastic masterpiece."},
    )
    assert resp.status_code == 200
    data = resp.json()
    total = data["probabilities"]["positive"] + data["probabilities"]["negative"]
    assert abs(total - 1.0) < 1e-4


def test_predict_model_unavailable_returns_503(client):
    """When model is not loaded, /predict should return 503."""
    with patch("app.model._model_loaded", False):
        resp = client.post("/predict", json={"text": "Great film!"})
    assert resp.status_code == 503


def test_predict_inference_time_is_positive(client_with_model):
    resp = client_with_model.post("/predict", json={"text": "Great film!"})
    assert resp.status_code == 200
    assert resp.json()["inference_time_ms"] > 0
