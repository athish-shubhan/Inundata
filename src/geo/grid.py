"""The analysis grid: one metric-CRS raster that every layer is warped onto."""
from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import array_bounds, rowcol
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform, transform_bounds

from src.config import CRS, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, METRIC_CRS, RESOLUTION_M


@dataclass(frozen=True)
class Grid:
    transform: Affine
    height: int
    width: int
    crs: str = METRIC_CRS

    @property
    def shape(self):
        return self.height, self.width

    @property
    def res(self) -> float:
        return self.transform.a

    def buffered(self, n_cells: int) -> "Grid":
        t = self.transform @ Affine.translation(-n_cells, -n_cells)
        return Grid(t, self.height + 2 * n_cells, self.width + 2 * n_cells, self.crs)

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates in the metric CRS, each shaped like the grid."""
        cols, rows = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        x = self.transform.c + cols * self.transform.a
        y = self.transform.f + rows * self.transform.e
        return x, y

    def lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.xy()
        lon, lat = transform(self.crs, CRS, x.ravel(), y.ravel())
        return np.asarray(lon).reshape(self.shape), np.asarray(lat).reshape(self.shape)

    def cell_of(self, lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row/col for WGS84 points, plus a mask of which points fall inside the grid."""
        x, y = transform(CRS, self.crs, np.atleast_1d(lon), np.atleast_1d(lat))
        r, c = rowcol(self.transform, x, y)
        r, c = np.asarray(r), np.asarray(c)
        inside = (r >= 0) & (r < self.height) & (c >= 0) & (c < self.width)
        return r, c, inside

    def to_wgs84(self, arr: np.ndarray, resampling=Resampling.nearest) -> tuple[np.ndarray, list[list[float]]]:
        """Warp a grid-shaped array to lat/lon for web maps; returns it with [[south, west], [north, east]]."""
        src_bounds = array_bounds(self.height, self.width, self.transform)
        tf, w, h = calculate_default_transform(self.crs, CRS, self.width, self.height, *src_bounds)
        dst = np.full((h, w), np.nan, dtype="float32")
        reproject(arr.astype("float32"), dst, src_transform=self.transform, src_crs=self.crs, src_nodata=np.nan,
                  dst_transform=tf, dst_crs=CRS, dst_nodata=np.nan, resampling=resampling)
        west, south, east, north = array_bounds(h, w, tf)
        return dst, [[south, west], [north, east]]


def analysis_grid() -> Grid:
    res = RESOLUTION_M
    left, bottom, right, top = transform_bounds(CRS, METRIC_CRS, LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    left, top = np.floor(left / res) * res, np.ceil(top / res) * res
    width = int(np.ceil((right - left) / res))
    height = int(np.ceil((top - bottom) / res))
    return Grid(Affine(res, 0, left, 0, -res, top), height, width)


def warp(path, grid: Grid, resampling=Resampling.bilinear, band=1) -> np.ndarray:
    dst = np.full(grid.shape, np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, band), dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=grid.transform, dst_crs=grid.crs,
                  dst_nodata=np.nan, resampling=resampling)
    return dst


def write_stack(path, grid: Grid, layers: dict[str, np.ndarray], tags: dict | None = None):
    with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width, count=len(layers),
                       dtype="float32", crs=grid.crs, transform=grid.transform, nodata=np.nan,
                       compress="deflate", predictor=3) as dst:
        for i, (name, arr) in enumerate(layers.items(), start=1):
            dst.write(arr.astype("float32"), i)
            dst.set_band_description(i, name)
        dst.update_tags(**(tags or {}))


def read_stack(path) -> tuple[Grid, dict[str, np.ndarray], dict]:
    with rasterio.open(path) as src:
        grid = Grid(src.transform, src.height, src.width, src.crs.to_string())
        layers = {name: src.read(i) for i, name in enumerate(src.descriptions, start=1)}
        return grid, layers, src.tags()
