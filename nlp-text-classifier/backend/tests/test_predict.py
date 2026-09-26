"""
test_predict.py — Tests for POST /predict/nlp-sentiment and POST /predict/vision-transformer.
"""

import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ── Mock data ─────────────────────────────────────────────────────────────────

NLP_MOCK_RESULT = {
    "model_id": "nlp-sentiment",
    "prediction": "positive",
    "confidence": 0.964123,
    "probabilities": {"positive": 0.964123, "negative": 0.035877},
    "inference_time_ms": 43.2,
}

VIT_MOCK_RESULT = {
    "model_id": "vision-transformer",
    "prediction": "cat",
    "confidence": 0.82,
    "top_predictions": [
        {"class": "cat",  "probability": 0.82},
        {"class": "dog",  "probability": 0.10},
        {"class": "deer", "probability": 0.05},
        {"class": "frog", "probability": 0.02},
        {"class": "bird", "probability": 0.01},
    ],
    "all_probabilities": {
        "airplane": 0.0, "automobile": 0.0, "bird": 0.01,
        "cat": 0.82, "deer": 0.05, "dog": 0.10,
        "frog": 0.02, "horse": 0.0, "ship": 0.0, "truck": 0.0,
    },
    "inference_time_ms": 120.5,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_registry(nlp_loaded=True, vit_loaded=True,
                   nlp_result=NLP_MOCK_RESULT, vit_result=VIT_MOCK_RESULT):
    """Build a mock ModelRegistry with configurable adapters."""
    from app.registry.registry import ModelRegistry

    nlp_adapter = MagicMock()
    nlp_adapter.is_loaded.return_value = nlp_loaded
    nlp_adapter.predict.return_value = nlp_result

    vit_adapter = MagicMock()
    vit_adapter.is_loaded.return_value = vit_loaded
    vit_adapter.predict.return_value = vit_result

    registry = MagicMock(spec=ModelRegistry)
    registry.any_loaded.return_value = nlp_loaded or vit_loaded

    def _get(model_id):
        return {"nlp-sentiment": nlp_adapter, "vision-transformer": vit_adapter}.get(model_id)

    registry.get.side_effect = _get
    registry.all.return_value = [nlp_adapter, vit_adapter]
    return registry


@pytest.fixture
def client():
    with patch("app.main.ModelRegistry", return_value=_make_registry()):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_nlp_only():
    with patch("app.main.ModelRegistry", return_value=_make_registry(vit_loaded=False)):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_no_models():
    with patch("app.main.ModelRegistry", return_value=_make_registry(nlp_loaded=False, vit_loaded=False)):
        from app.main import app
        with TestClient(app) as c:
            yield c


# ── NLP tests ─────────────────────────────────────────────────────────────────

def test_nlp_predict_returns_200(client):
    resp = client.post("/predict/nlp-sentiment", json={"text": "Brilliant film!"})
    assert resp.status_code == 200


def test_nlp_predict_response_shape(client):
    resp = client.post("/predict/nlp-sentiment", json={"text": "Brilliant film!"})
    data = resp.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "positive" in data["probabilities"]
    assert "negative" in data["probabilities"]
    assert "inference_time_ms" in data


def test_nlp_predict_probabilities_sum_to_one(client):
    resp = client.post("/predict/nlp-sentiment", json={"text": "Brilliant!"})
    data = resp.json()
    total = data["probabilities"]["positive"] + data["probabilities"]["negative"]
    assert abs(total - 1.0) < 1e-4


def test_nlp_predict_inference_time_positive(client):
    resp = client.post("/predict/nlp-sentiment", json={"text": "Great film!"})
    assert resp.json()["inference_time_ms"] > 0


def test_nlp_predict_503_when_model_unavailable(client_no_models):
    resp = client_no_models.post("/predict/nlp-sentiment", json={"text": "Great film!"})
    assert resp.status_code == 503


def test_legacy_predict_endpoint_still_works(client):
    """POST /predict (old endpoint) should still route to NLP sentiment."""
    resp = client.post("/predict", json={"text": "Great film!"})
    assert resp.status_code == 200


# ── ViT tests ─────────────────────────────────────────────────────────────────

def _png_bytes():
    """Return minimal valid 1x1 PNG bytes."""
    import base64
    # Tiny 1x1 red PNG
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
        "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
    )


def test_vit_predict_returns_200(client):
    resp = client.post(
        "/predict/vision-transformer",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200


def test_vit_predict_response_shape(client):
    resp = client.post(
        "/predict/vision-transformer",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    data = resp.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "top_predictions" in data
    assert isinstance(data["top_predictions"], list)
    assert len(data["top_predictions"]) > 0
    assert "inference_time_ms" in data


def test_vit_predict_rejects_non_image(client):
    resp = client.post(
        "/predict/vision-transformer",
        files={"file": ("doc.pdf", b"fake pdf content", "application/pdf")},
    )
    assert resp.status_code == 415


def test_vit_predict_503_when_model_unavailable(client_nlp_only):
    resp = client_nlp_only.post(
        "/predict/vision-transformer",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 503


def test_predict_unknown_model_returns_404(client):
    resp = client.post("/predict/unknown-model", json={"text": "hello"})
    assert resp.status_code == 404
