"""
test_health.py — Tests for GET / and GET /health.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ── Shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """TestClient with both model adapters mocked as unloaded."""
    from app.registry.registry import ModelRegistry

    mock_registry = MagicMock(spec=ModelRegistry)
    mock_registry.any_loaded.return_value = False
    mock_registry.all.return_value = []

    with patch("app.main.ModelRegistry", return_value=mock_registry):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_loaded():
    """TestClient where at least one model reports as loaded."""
    from app.registry.registry import ModelRegistry

    mock_registry = MagicMock(spec=ModelRegistry)
    mock_registry.any_loaded.return_value = True
    mock_registry.all.return_value = []

    with patch("app.main.ModelRegistry", return_value=mock_registry):
        from app.main import app
        with TestClient(app) as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "AI Model Lab API"
    assert data["version"] == "2.0.0"
    assert data["status"] == "running"


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert isinstance(data["model_loaded"], bool)


def test_health_model_loaded_false_when_no_models(client):
    resp = client.get("/health")
    assert resp.json()["model_loaded"] is False


def test_health_model_loaded_true_when_model_available(client_loaded):
    resp = client_loaded.get("/health")
    assert resp.json()["model_loaded"] is True


def test_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_redoc_available(client):
    resp = client.get("/redoc")
    assert resp.status_code == 200
