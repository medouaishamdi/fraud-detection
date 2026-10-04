"""API tests. Run from the project root with:  pytest"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from api import app  # noqa: E402

client = TestClient(app)

NORMAL = {
    "amount": 300.0, "transaction_hour": 14, "distance_from_home": 5.0,
    "merchant_risk": 0.10, "device_trusted": 1, "previous_transactions": 15,
}
SUSPICIOUS = {
    "amount": 1800.0, "transaction_hour": 3, "distance_from_home": 400.0,
    "merchant_risk": 0.90, "device_trusted": 0, "previous_transactions": 1,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_normal_transaction_is_low_risk():
    body = client.post("/predict", json=NORMAL).json()
    assert body["is_fraud"] is False
    assert body["risk_level"] == "LOW"


def test_suspicious_transaction_is_flagged():
    body = client.post("/predict", json=SUSPICIOUS).json()
    assert body["is_fraud"] is True
    assert body["risk_level"] == "HIGH"


def test_invalid_input_is_rejected():
    bad = {**NORMAL, "transaction_hour": 30}
    assert client.post("/predict", json=bad).status_code == 422
