import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.warp import Resampling
from scipy.ndimage import distance_transform_edt, uniform_filter

from src.config import DATA_RAW
from src.geo.grid import Grid, warp

DEM_PATH = DATA_RAW / "terrain" / "copernicus_dem30.tif"
RIVER_PATH = DATA_RAW / "gis" / "river.geojson"


def load_dem(grid: Grid) -> np.ndarray:
    return warp(DEM_PATH, grid, Resampling.bilinear)


def slope_deg(elev: np.ndarray, res: float) -> np.ndarray:
    gy, gx = np.gradient(elev, res)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")


def topographic_position(elev: np.ndarray, res: float, radius_m: float = 300) -> np.ndarray:
    """Elevation minus the local mean: negative in valley floors, positive on ridges."""
    size = int(2 * radius_m / res) + 1
    return (elev - uniform_filter(elev, size=size, mode="nearest")).astype("float32")


def river_mask(grid: Grid) -> np.ndarray:
    lines = gpd.read_file(RIVER_PATH).to_crs(grid.crs).geometry
    return rasterize(((g, 1) for g in lines), out_shape=grid.shape, transform=grid.transform,
                     all_touched=True, dtype="uint8").astype(bool)


def distance_to(mask: np.ndarray, res: float) -> np.ndarray:
    return (distance_transform_edt(~mask) * res).astype("float32")
