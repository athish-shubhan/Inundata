from fastapi.testclient import TestClient

from src.api.main import app
from src.config import DATA_RAW

client = TestClient(app)
DOWNTOWN = {"lat": 32.2158, "lon": 130.7535}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_ready"] is True


def test_region():
    r = client.get("/region")
    assert r.status_code == 200
    assert "bounds" in r.json() and "downtown" in r.json()["known_places"]


def test_predict():
    r = client.get("/predict", params=DOWNTOWN)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1
    assert "flooded" in body["observed_2020"]


def test_predict_outside_aoi():
    assert client.get("/predict", params={"lat": 35.0, "lon": 139.0}).status_code == 404


def test_predict_missing_params():
    assert client.get("/predict").status_code == 422


def test_exposed_assets():
    r = client.get("/assets/exposed", params=DOWNTOWN | {"radius_km": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["n_assets"] == len(body["assets"]) > 0
    scores = [a["risk_score"] for a in body["assets"] if a["risk_score"] is not None]
    assert scores == sorted(scores, reverse=True)


def test_exposed_assets_radius_validated():
    assert client.get("/assets/exposed", params=DOWNTOWN | {"radius_km": -1}).status_code == 422


def test_exposed_infrastructure():
    r = client.get("/infrastructure/exposed", params=DOWNTOWN | {"radius_km": 2})
    assert r.status_code == 200
    assert r.json()["n_sites"] > 0


def test_rainfall():
    r = client.get("/rainfall")
    assert r.status_code == 200
    assert r.json()["points"]


def test_model_metrics():
    r = client.get("/model/metrics")
    assert r.status_code == 200
    assert {"selected_model", "spatial_cv", "random_cv", "caveat"} <= set(r.json())


def test_grid():
    r = client.get("/grid", params={"step": 20})
    assert r.status_code == 200
    assert 100 < len(r.json()) < 1000


def test_agent_analyze():
    r = client.post("/agent/analyze", json={"query": "flood risk downtown within 1km"})
    assert r.status_code == 200
    assert "report_markdown" in r.json()


def test_agent_analyze_query_too_short():
    assert client.post("/agent/analyze", json={"query": "ab"}).status_code == 422


def test_ingest_csv():
    with open(DATA_RAW / "client" / "client_portfolio.csv", "rb") as f:
        r = client.post("/ingest", files={"file": ("client_portfolio.csv", f, "text/csv")})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["n_rows"] == 120


def test_ingest_bad_format():
    r = client.post("/ingest", files={"file": ("bad.xyz", b"junk", "application/octet-stream")})
    assert r.status_code == 400
