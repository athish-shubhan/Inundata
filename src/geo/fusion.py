"""Builds data/<aoi>/processed/features.tif: every layer on one 30 m metric grid.

The stack holds three kinds of band:
  * predictors, all observable before the event (terrain, hydrology, pre-flood imagery);
  * the reference label, from the GSI inundation estimate;
  * diagnostics (permanent water, cloud, the old NDWI-change proxy) used for
    masking and for measuring how good the proxy label was.
"""
import time

import numpy as np
from rasterio.warp import Resampling

from src.config import DATA_PROC, DATA_RAW
from src.geo.grid import analysis_grid, warp, write_stack
from src.geo.hydrology import hand_from_dem
from src.geo.indices import cloud_mask, spectral_indices
from src.geo.terrain import distance_to, load_dem, river_mask, slope_deg, topographic_position

ROUTING_BUFFER_CELLS = 100      # 3 km of extra DEM around the AOI so flow paths are not cut at the edge
FLOODED_FRACTION = 0.5          # a cell is flooded if at least half its area is inside the GSI extent
PROXY_NDWI_CHANGE = 0.04        # the original proxy label: NDWI(post) - NDWI(pre) above this
PERMANENT_WATER_NDWI = 0.0      # open water before the event is river channel, not flood
REFERENCE_PATH = DATA_RAW / "reference" / "gsi_inundation.tif"


def build_features() -> dict:
    t0 = time.perf_counter()
    grid = analysis_grid()
    big = grid.buffered(ROUTING_BUFFER_CELLS)
    inner = (slice(ROUTING_BUFFER_CELLS, -ROUTING_BUFFER_CELLS),) * 2

    dem_big = load_dem(big)
    hyd = hand_from_dem(dem_big, big.res, river_mask(big))
    elev = dem_big[inner]
    river = river_mask(grid)

    pre = spectral_indices("pre", grid)
    post = spectral_indices("post", grid)
    flood_frac = warp(REFERENCE_PATH, grid, Resampling.average)

    layers = {
        "elev_m": elev,
        "slope_deg": slope_deg(elev, grid.res),
        "hand_m": hyd["hand"][inner],
        "river_dist_m": distance_to(river, grid.res),
        "tpi_m": topographic_position(elev, grid.res),
        "log_flow_acc": np.log10(hyd["flow_acc"][inner]),
        "ndvi_pre": pre["ndvi"],
        "ndwi_pre": pre["ndwi"],
        "ndbi_pre": pre["ndbi"],
        "flood_fraction": flood_frac,
        "flooded": (flood_frac >= FLOODED_FRACTION).astype("float32"),
        "permanent_water": (pre["ndwi"] > PERMANENT_WATER_NDWI).astype("float32"),
        "cloud": (cloud_mask("pre", grid) | cloud_mask("post", grid)).astype("float32"),
        "ndwi_change": post["ndwi"] - pre["ndwi"],
        "proxy_label": ((post["ndwi"] - pre["ndwi"]) > PROXY_NDWI_CHANGE).astype("float32"),
    }
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    write_stack(DATA_PROC / "features.tif", grid, layers, tags={"flooded_fraction_threshold": FLOODED_FRACTION})
    return {"shape": grid.shape, "n_cells": grid.height * grid.width, "seconds": round(time.perf_counter() - t0, 1),
            "flooded_cells": int(layers["flooded"].sum())}


if __name__ == "__main__":
    print(build_features())
