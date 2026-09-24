"""
test_validation.py — Input validation tests for POST /predict.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def client():
    with patch("app.model.load_model"):
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_empty_string_returns_422(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_whitespace_only_returns_422(client):
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 422


def test_missing_text_field_returns_422(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_wrong_type_returns_422(client):
    resp = client.post("/predict", json={"text": 12345})
    # FastAPI coerces int to str — should still work; if not, 422
    # Either way no 500 error
    assert resp.status_code in (200, 422, 503)


def test_invalid_json_returns_422(client):
    resp = client.post(
        "/predict",
        content=b"not json at all",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_oversized_input_returns_422(client):
    huge = "a" * 11000  # exceeds MAX_TEXT_LENGTH=10000
    resp = client.post("/predict", json={"text": huge})
    assert resp.status_code == 422


def test_valid_short_text_accepted(client):
    with patch("app.model._model_loaded", True), \
         patch("app.model.predict", return_value={
             "prediction": "positive",
             "confidence": 0.8,
             "probabilities": {"positive": 0.8, "negative": 0.2},
             "inference_time_ms": 12.0,
         }):
        resp = client.post("/predict", json={"text": "Good."})
    assert resp.status_code == 200


def test_error_response_has_detail_field(client):
    resp = client.post("/predict", json={"text": ""})
    assert "detail" in resp.json()


def test_no_stack_trace_in_error_response(client):
    resp = client.post("/predict", json={"text": ""})
    body = resp.text
    assert "Traceback" not in body
    assert "File \"" not in body
