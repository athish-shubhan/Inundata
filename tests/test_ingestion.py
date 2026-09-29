import pandas as pd
import pytest
from src.config import DATA_RAW
from src.ingestion.csv_source import load_csv
from src.ingestion.geojson_source import load_geojson
from src.ingestion.client_adapter import ingest_client_dataset

def test_load_csv_valid(tmp_path):
    p = tmp_path / "assets.csv"
    pd.DataFrame({"id": [1, 2], "lat": [32.8, 32.81], "lon": [130.75, 130.76], "value": [100, 200]}).to_csv(p, index=False)
    res = load_csv(p, id_col="id")
    assert res.report.ok
    assert len(res.gdf) == 2
    assert res.gdf.crs is not None

def test_load_csv_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"id": [1], "value": [100]}).to_csv(p, index=False)
    res = load_csv(p, id_col="id")
    assert not res.report.ok
    assert res.gdf is None

def test_load_csv_duplicate_and_missing_warn(tmp_path):
    p = tmp_path / "dup.csv"
    pd.DataFrame({"id": [1, 1], "lat": [32.8, None], "lon": [130.75, 130.76]}).to_csv(p, index=False)
    res = load_csv(p, id_col="id")
    assert res.report.ok
    assert any("duplicate" in w for w in res.report.warnings)
    assert len(res.gdf) == 1

def test_load_geojson_river():
    res = load_geojson(DATA_RAW / "gis" / "kuma_river.geojson")
    assert res.report.ok
    assert res.report.n_rows == 1
    assert "LineString" in res.report.meta["geom_types"]

def test_client_adapter_unsupported_format(tmp_path):
    p = tmp_path / "data.xyz"
    p.write_text("junk")
    with pytest.raises(ValueError):
        ingest_client_dataset(p)

def test_client_adapter_csv_roundtrip():
    result = ingest_client_dataset(DATA_RAW / "client" / "client_portfolio.csv")
    assert result.report.ok
    assert len(result.gdf) == 120

def test_client_adapter_geojson_ignores_csv_only_kwargs():
    result = ingest_client_dataset(DATA_RAW / "gis" / "kuma_river.geojson", lat_col="lat", lon_col="lon")
    assert result.report.ok
    assert result.report.n_rows == 1
