import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from src.config import DATA_RAW, GRID_STEP, CRS, METRIC_CRS

def river_distance_m(lats, lons):
    river = gpd.read_file(DATA_RAW / "gis" / "kuma_river.geojson").to_crs(METRIC_CRS)
    line = river.geometry.iloc[0]
    lon_g, lat_g = np.meshgrid(lons, lats)
    pts = gpd.GeoSeries(gpd.points_from_xy(lon_g.ravel(), lat_g.ravel()), crs=CRS).to_crs(METRIC_CRS)
    dist = pts.distance(line).values.reshape(lat_g.shape)
    return dist.astype("float32")

def height_above_drainage(elev, river_dist):
    near = river_dist <= np.nanpercentile(river_dist, 8)
    drainage_elev = np.nanmin(elev[near]) if near.any() else np.nanmin(elev)
    return (elev - drainage_elev).astype("float32")

def _read_resampled(path, lats, lons):
    dst_tf = from_origin(lons[0] - GRID_STEP / 2, lats[0] + GRID_STEP / 2, GRID_STEP, GRID_STEP)
    dst = np.full((len(lats), len(lons)), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst, src_transform=src.transform, src_crs=src.crs,
                   dst_transform=dst_tf, dst_crs=CRS, resampling=Resampling.bilinear)
    return dst

def band_index(scene_label, band, lats, lons):
    p = DATA_RAW / "satellite" / f"{scene_label}_{band}.tif"
    return _read_resampled(p, lats, lons)

def ndwi(scene_label, lats, lons):
    g = band_index(scene_label, "B03", lats, lons)
    n = band_index(scene_label, "B08", lats, lons)
    return (g - n) / (g + n + 1e-6)

def ndvi(scene_label, lats, lons):
    n = band_index(scene_label, "B08", lats, lons)
    r = band_index(scene_label, "B04", lats, lons)
    return (n - r) / (n + r + 1e-6)

def cloud_mask(scene_label, lats, lons):
    scl = band_index(scene_label, "SCL", lats, lons)
    return np.isin(np.round(scl), [3, 8, 9, 10])
