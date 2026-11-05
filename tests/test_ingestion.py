import pandas as pd
import pytest
from src.config import DATA_RAW
from src.ingestion.csv_source import load_csv
from src.ingestion.geojson_source import load_geojson
from src.ingestion.client_adapter import ingest_client_dataset

def test_load_csv_valid(tmp_path):
    p = tmp_path / "assets.csv"
    pd.DataFrame({"id": [1, 2], "lat": [32.21, 32.22], "lon": [130.75, 130.76], "value": [100, 200]}).to_csv(p, index=False)
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
    pd.DataFrame({"id": [1, 1], "lat": [32.21, None], "lon": [130.75, 130.76]}).to_csv(p, index=False)
    res = load_csv(p, id_col="id")
    assert res.report.ok
    assert any("duplicate" in w for w in res.report.warnings)
    assert len(res.gdf) == 1

def test_load_geojson_river():
    res = load_geojson(DATA_RAW / "gis" / "river.geojson")
    assert res.report.ok
    assert res.report.n_rows > 1
    assert res.report.meta["geom_types"] == ["LineString"]

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
    result = ingest_client_dataset(DATA_RAW / "gis" / "infrastructure.geojson", lat_col="lat", lon_col="lon")
    assert result.report.ok
    assert result.report.meta["geom_types"] == ["Point"]

def test_load_csv_rejects_out_of_range_coordinates(tmp_path):
    # Swapped columns put longitude 130.75 in the latitude field.
    p = tmp_path / "swapped.csv"
    pd.DataFrame({"id": [1, 2], "lat": [130.75, 32.2], "lon": [32.21, 130.7]}).to_csv(p, index=False)
    res = load_csv(p)
    assert not res.report.ok and res.gdf is None
    assert "outside lat [-90, 90]" in res.report.errors[0]

def test_load_csv_drops_non_numeric_coordinates(tmp_path):
    p = tmp_path / "text.csv"
    p.write_text("id,lat,lon\n1,abc,130.75\n2,32.2,130.7\n")
    res = load_csv(p)
    assert res.report.ok
    assert res.report.warnings == ["1 non-numeric values in 'lat'"]
    assert len(res.gdf) == 1 and res.report.meta["n_valid"] == 1

def _write_tif(path, arr, nodata):
    import rasterio
    from affine import Affine
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs="EPSG:4326", transform=Affine(0.01, 0, 130.7, 0, -0.01, 32.3),
                       nodata=nodata) as ds:
        ds.write(arr, 1)

def test_load_raster_counts_nan_nodata(tmp_path):
    import numpy as np
    arr = np.ones((4, 5), "float32")
    arr[0, :2] = np.nan
    _write_tif(tmp_path / "nan.tif", arr, nodata=np.nan)
    res = ingest_client_dataset(tmp_path / "nan.tif")
    assert res.gdf is None and res.report.ok
    assert res.report.warnings == ["2 nodata pixels"]
    assert res.report.n_rows == 20 and res.report.meta["shape"] == [4, 5]

def test_load_raster_counts_integer_nodata(tmp_path):
    import numpy as np
    arr = np.zeros((3, 3), "int16")
    arr[1, 1] = -9999
    _write_tif(tmp_path / "int.tif", arr, nodata=-9999)
    assert ingest_client_dataset(tmp_path / "int.tif").report.warnings == ["1 nodata pixels"]

def test_load_raster_unreadable(tmp_path):
    p = tmp_path / "broken.tif"
    p.write_bytes(b"not a tiff")
    res = ingest_client_dataset(p)
    assert not res.report.ok and "failed to read raster" in res.report.errors[0]
