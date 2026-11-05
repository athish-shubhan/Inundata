import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .base import IngestReport, IngestResult
from .validate import check_coordinates, check_crs, check_duplicates, check_geometry, check_missing


def load_csv(path, lat_col="lat", lon_col="lon", id_col=None, required: list[str] | None = None) -> IngestResult:
    """Load a point CSV in WGS84.

    Rows with missing or non-numeric coordinates are dropped with a warning; coordinates
    outside lat [-90, 90] / lon [-180, 180] are an error (usually swapped or projected columns).
    """
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

    df = check_coordinates(df, lat_col, lon_col, rep)
    if rep.errors:
        return IngestResult(None, rep)

    geom = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    check_geometry(gdf, rep)
    check_crs(gdf, rep)
    rep.meta["columns"] = list(df.columns)
    rep.meta["n_valid"] = len(gdf)
    return IngestResult(gdf, rep)
