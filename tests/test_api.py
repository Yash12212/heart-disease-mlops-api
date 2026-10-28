"""Automated tests for the Heart Disease Prediction API."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """The health endpoint should report that the service is up."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_returns_valid_response():
    """A valid patient payload should return a well formed prediction."""
    payload = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["label"] in ["Disease", "No disease"]
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_invalid_input():
    """Out of range values should be rejected with a 422 status code."""
    payload = {
        "age": 500,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
