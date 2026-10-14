"""
FraudShield - API Endpoints Unit Tests
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["feature_count"] == 30
    assert "required_features" in data
    assert len(data["required_features"]) == 30
