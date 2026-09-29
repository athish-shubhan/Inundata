import geopandas as gpd
from .base import IngestReport, IngestResult
from .validate import check_geometry, check_crs, check_missing

def load_geojson(path, required: list[str] | None = None) -> IngestResult:
    rep = IngestReport(source=str(path))
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        rep.errors.append(f"failed to read geojson: {e}")
        return IngestResult(None, rep)

    rep.n_rows = len(gdf)
    if required:
        check_missing(gdf, required, rep)
    check_geometry(gdf, rep)
    check_crs(gdf, rep)
    rep.meta["columns"] = [c for c in gdf.columns if c != "geometry"]
    rep.meta["geom_types"] = gdf.geometry.geom_type.unique().tolist()
    return IngestResult(gdf, rep)

load_shapefile = load_geojson
