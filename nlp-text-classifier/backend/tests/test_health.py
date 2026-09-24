"""
test_health.py — Tests for GET /health and GET /
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def client():
    with patch("app.model.load_model"):  # skip actual TF load in tests
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "NLP Text Classifier"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "healthy"
    assert isinstance(data["model_loaded"], bool)


def test_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_redoc_available(client):
    resp = client.get("/redoc")
    assert resp.status_code == 200
