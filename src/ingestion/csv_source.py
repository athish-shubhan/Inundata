import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from .base import IngestReport, IngestResult
from .validate import check_missing, check_duplicates, check_geometry, check_crs

def load_csv(path, lat_col="lat", lon_col="lon", id_col=None, required: list[str] | None = None) -> IngestResult:
    rep = IngestReport(source=str(path))
    try:
        df = pd.read_csv(path)
    except Exception as e:
        rep.errors.append(f"failed to read csv: {e}")
        return IngestResult(None, rep)

    rep.n_rows = len(df)
    check_missing(df, (required or []) + [lat_col, lon_col], rep)
    if id_col:
        check_duplicates(df, id_col, rep)
    if rep.errors:
        return IngestResult(None, rep)

    df = df.dropna(subset=[lat_col, lon_col])
    geom = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    check_geometry(gdf, rep)
    check_crs(gdf, rep)
    rep.meta["columns"] = list(df.columns)
    rep.meta["n_valid"] = len(gdf)
    return IngestResult(gdf, rep)
