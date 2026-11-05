import geopandas as gpd
import pandas as pd
from .base import IngestReport

def check_missing(df, cols, rep: IngestReport):
    for c in cols:
        if c not in df.columns:
            rep.errors.append(f"missing required column: {c}")
            continue
        n = df[c].isna().sum()
        if n:
            rep.warnings.append(f"{n} missing values in '{c}'")

def check_coordinates(df, lat_col: str, lon_col: str, rep: IngestReport):
    """Coerce lat/lon to numbers and drop rows without a usable coordinate.

    Missing values were already reported by check_missing; non-numeric values are
    reported here as a warning. Out-of-range values are an error. Returns the kept rows.
    """
    df = df.copy()
    for col in (lat_col, lon_col):
        num = pd.to_numeric(df[col], errors="coerce")
        n_bad = int((num.isna() & df[col].notna()).sum())
        if n_bad:
            rep.warnings.append(f"{n_bad} non-numeric values in '{col}'")
        df[col] = num
    df = df.dropna(subset=[lat_col, lon_col])
    n_out = int(((df[lat_col].abs() > 90) | (df[lon_col].abs() > 180)).sum())
    if n_out:
        rep.errors.append(f"{n_out} rows with coordinates outside lat [-90, 90] / lon [-180, 180]; "
                          f"check that '{lat_col}' and '{lon_col}' are WGS84 degrees and not swapped")
    return df

def check_duplicates(df, key, rep: IngestReport):
    if key not in df.columns:
        return
    n = df[key].duplicated().sum()
    if n:
        rep.warnings.append(f"{n} duplicate values in '{key}'")

def check_geometry(gdf: gpd.GeoDataFrame, rep: IngestReport):
    n_null = gdf.geometry.isna().sum()
    if n_null:
        rep.errors.append(f"{n_null} rows with null geometry")
    n_invalid = (~gdf.geometry.is_valid).sum()
    if n_invalid:
        rep.warnings.append(f"{n_invalid} invalid geometries (auto-fixed with buffer(0))")
        gdf.loc[~gdf.geometry.is_valid, "geometry"] = gdf.loc[~gdf.geometry.is_valid, "geometry"].buffer(0)

def check_crs(gdf: gpd.GeoDataFrame, rep: IngestReport, target="EPSG:4326"):
    if gdf.crs is None:
        rep.warnings.append(f"no CRS set, assuming {target}")
        gdf.set_crs(target, inplace=True)
    elif str(gdf.crs) != target:
        rep.meta["reprojected_from"] = str(gdf.crs)
        gdf.to_crs(target, inplace=True)

def check_bounds(gdf: gpd.GeoDataFrame, bbox, rep: IngestReport):
    lon_min, lat_min, lon_max, lat_max = bbox
    b = gdf.total_bounds
    if b[0] < lon_min - 1 or b[2] > lon_max + 1 or b[1] < lat_min - 1 or b[3] > lat_max + 1:
        rep.warnings.append(f"data bounds {list(b)} fall well outside expected AOI {bbox}")
