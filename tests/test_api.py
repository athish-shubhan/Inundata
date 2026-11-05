import pytest
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


def test_ingest_csv_out_of_range_reported_not_ok():
    r = client.post("/ingest", files={"file": ("swapped.csv", b"id,lat,lon\n1,130.75,32.21\n", "text/csv")})
    assert r.status_code == 200
    assert r.json()["ok"] is False and "outside lat" in r.json()["errors"][0]


def test_ingest_raster():
    with open(DATA_RAW / "terrain" / "copernicus_dem30.tif", "rb") as f:
        r = client.post("/ingest", files={"file": ("copernicus_dem30.tif", f, "image/tiff")})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "copernicus_dem30.tif" and body["meta"]["crs"]


@pytest.mark.parametrize("params", [{"lat": 1000, "lon": 130.75}, {"lat": 32.2, "lon": -500},
                                    {"lat": "nan", "lon": "nan"}, {"lat": "inf", "lon": 130.7}])
def test_predict_rejects_invalid_coordinates(params):
    r = client.get("/predict", params=params)
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("path", ["/assets/exposed", "/infrastructure/exposed"])
def test_exposure_rejects_invalid_coordinates_and_radius(path):
    assert client.get(path, params={"lat": 95, "lon": 130.75}).status_code == 422
    assert client.get(path, params=DOWNTOWN | {"radius_km": 21}).status_code == 422
    assert client.get(path, params=DOWNTOWN | {"radius_km": 20}).status_code == 200


def test_ingest_rejects_oversized_upload(monkeypatch, tmp_path):
    import src.api.main as api_main
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(api_main, "UPLOAD_CHUNK_BYTES", 256)
    monkeypatch.setattr(api_main.tempfile, "tempdir", str(tmp_path))
    body = b"id,lat,lon\n" + b"1,32.2,130.7\n" * 200  # 2.6 KB
    r = client.post("/ingest", files={"file": ("big.csv", body, "text/csv")})
    assert r.status_code == 413 and "larger than" in r.json()["detail"]
    assert list(tmp_path.iterdir()) == []  # the partial temp file is removed
    small = client.post("/ingest", files={"file": ("small.csv", body[:500], "text/csv")})
    assert small.status_code == 200 and small.json()["ok"] is True


def test_ingest_rejects_unsupported_format_before_reading(monkeypatch, tmp_path):
    import src.api.main as api_main
    monkeypatch.setattr(api_main.tempfile, "tempdir", str(tmp_path))
    r = client.post("/ingest", files={"file": ("notes.docx", b"x" * 10, "application/octet-stream")})
    assert r.status_code == 400 and "unsupported format '.docx'" in r.json()["detail"]
    assert client.post("/ingest", files={"file": ("noext", b"x", "text/plain")}).status_code == 400
    assert list(tmp_path.iterdir()) == []
