"""
test_validation.py — Input validation tests for NLP and ViT endpoints.
"""

import base64
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from tests.test_predict import _make_registry, NLP_MOCK_RESULT


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    with patch("app.main.ModelRegistry", return_value=_make_registry()):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_no_models():
    with patch("app.main.ModelRegistry", return_value=_make_registry(nlp_loaded=False, vit_loaded=False)):
        from app.main import app
        with TestClient(app) as c:
            yield c


def _png_bytes():
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
        "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
    )


# ── NLP input validation ──────────────────────────────────────────────────────

def test_empty_string_returns_422(client):
    resp = client.post("/predict/nlp-sentiment", json={"text": ""})
    assert resp.status_code == 422


def test_whitespace_only_returns_422(client):
    resp = client.post("/predict/nlp-sentiment", json={"text": "   "})
    assert resp.status_code == 422


def test_missing_text_field_returns_422(client):
    resp = client.post("/predict/nlp-sentiment", json={})
    assert resp.status_code == 422


def test_invalid_json_returns_422(client):
    resp = client.post(
        "/predict/nlp-sentiment",
        content=b"not json at all",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_oversized_text_returns_422(client):
    huge = "a" * 11000  # exceeds MAX_TEXT_LENGTH=10000
    resp = client.post("/predict/nlp-sentiment", json={"text": huge})
    assert resp.status_code == 422


def test_valid_short_text_accepted(client):
    resp = client.post("/predict/nlp-sentiment", json={"text": "Good."})
    assert resp.status_code == 200


def test_error_response_has_detail_field(client):
    resp = client.post("/predict/nlp-sentiment", json={"text": ""})
    assert "detail" in resp.json()


def test_no_stack_trace_in_error_response(client):
    resp = client.post("/predict/nlp-sentiment", json={"text": ""})
    body = resp.text
    assert "Traceback" not in body
    assert 'File "' not in body


# ── ViT input validation ──────────────────────────────────────────────────────

def test_vit_rejects_pdf(client):
    resp = client.post(
        "/predict/vision-transformer",
        files={"file": ("doc.pdf", b"%PDF fake", "application/pdf")},
    )
    assert resp.status_code == 415


def test_vit_rejects_text_file(client):
    resp = client.post(
        "/predict/vision-transformer",
        files={"file": ("note.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 415


def test_vit_rejects_oversized_image(client):
    big_data = b"x" * (6 * 1024 * 1024)  # 6 MB > 5 MB limit
    resp = client.post(
        "/predict/vision-transformer",
        files={"file": ("big.png", big_data, "image/png")},
    )
    assert resp.status_code == 413


def test_vit_accepts_valid_png(client):
    resp = client.post(
        "/predict/vision-transformer",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200


def test_vit_accepts_jpeg(client):
    resp = client.post(
        "/predict/vision-transformer",
        files={"file": ("test.jpg", _png_bytes(), "image/jpeg")},
    )
    # Our mock doesn't actually decode the image, so 200 is expected
    assert resp.status_code == 200


# ── Legacy endpoint validation ────────────────────────────────────────────────

def test_legacy_endpoint_empty_text_returns_422(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_legacy_endpoint_valid_text_returns_200(client):
    resp = client.post("/predict", json={"text": "Brilliant film!"})
    assert resp.status_code == 200
