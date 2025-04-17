"""FastAPI/tests/test_app.py"""
import pytest
from fastapi.testclient import TestClient
from FastAPI.app import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Welcome" in r.json()["message"]

@pytest.mark.parametrize("fname,expected_class", [
    ("grass.jpg", "grass"),
    ("dandelion.jpg", "dandelion"),
])
def test_predict(fname, expected_class):
    # load a small sample image fixture under tests/fixtures/
    with open(f"tests/fixtures/{fname}", "rb") as f:
        files = {"file": (fname, f, "image/jpeg")}
        r = client.post("/predict", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["predicted_class"] == expected_class
    assert 0.0 <= data["confidence"] <= 1.0
