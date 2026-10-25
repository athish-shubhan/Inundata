import numpy as np
from rasterio.warp import Resampling

from src.config import DATA_RAW
from src.geo.grid import Grid, warp

CLOUD_SCL = [3, 8, 9, 10]  # cloud shadow, medium/high cloud, thin cirrus


def band(scene: str, name: str, grid: Grid) -> np.ndarray:
    # Average resampling: a 30 m cell should hold the mean of its nine 10 m pixels, not one interpolated sample.
    return warp(DATA_RAW / "satellite" / f"{scene}_{name}.tif", grid, Resampling.average)


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((a - b) / (a + b)).astype("float32")


def spectral_indices(scene: str, grid: Grid) -> dict[str, np.ndarray]:
    green, red, nir, swir = (band(scene, b, grid) for b in ("B03", "B04", "B08", "B11"))
    return {"ndwi": normalized_difference(green, nir),   # open water (McFeeters 1996)
            "ndvi": normalized_difference(nir, red),     # vegetation
            "ndbi": normalized_difference(swir, nir)}    # built-up surfaces


def cloud_mask(scene: str, grid: Grid) -> np.ndarray:
    scl = warp(DATA_RAW / "satellite" / f"{scene}_SCL.tif", grid, Resampling.nearest)
    return np.isin(scl, CLOUD_SCL)
