from pathlib import Path
from .csv_source import load_csv
from .geojson_source import load_geojson
from .raster_source import load_raster

LOADERS = {"csv": load_csv, "geojson": load_geojson, "json": load_geojson, "shp": load_geojson,
           "tif": load_raster, "tiff": load_raster}
CSV_KWARGS = {"lat_col", "lon_col", "id_col", "required"}

def detect_type(path) -> str:
    return Path(path).suffix.lstrip(".").lower()

def ingest_client_dataset(path, **kwargs):
    ext = detect_type(path)
    fn = LOADERS.get(ext)
    if fn is None:
        raise ValueError(f"unsupported format '.{ext}'. supported: {list(LOADERS)}")
    if fn is load_raster:
        return fn(path)
    if fn is load_csv:
        return fn(path, **kwargs)
    return fn(path, **{k: v for k, v in kwargs.items() if k not in CSV_KWARGS})
