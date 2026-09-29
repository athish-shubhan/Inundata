from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_ready"] is True

def test_region():
    r = client.get("/region")
    assert r.status_code == 200
    assert "bounds" in r.json()

def test_predict():
    r = client.get("/predict", params={"lat": 32.795, "lon": 130.755})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1

def test_predict_missing_params():
    r = client.get("/predict")
    assert r.status_code == 422

def test_exposed_assets():
    r = client.get("/assets/exposed", params={"lat": 32.795, "lon": 130.755, "radius_km": 2})
    assert r.status_code == 200
    assert "assets" in r.json()

def test_model_metrics():
    r = client.get("/model/metrics")
    assert r.status_code == 200
    assert "cv_metrics" in r.json()

def test_grid():
    r = client.get("/grid")
    assert r.status_code == 200
    assert len(r.json()) > 100

def test_agent_analyze():
    r = client.post("/agent/analyze", json={"query": "flood risk downtown within 1km"})
    assert r.status_code == 200
    assert "report_markdown" in r.json()

def test_agent_analyze_query_too_short():
    r = client.post("/agent/analyze", json={"query": "ab"})
    assert r.status_code == 422

def test_ingest_csv():
    with open("data/raw/client/client_portfolio.csv", "rb") as f:
        r = client.post("/ingest", files={"file": ("client_portfolio.csv", f, "text/csv")})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["n_rows"] == 120

def test_ingest_bad_format():
    r = client.post("/ingest", files={"file": ("bad.xyz", b"junk", "application/octet-stream")})
    assert r.status_code == 400
