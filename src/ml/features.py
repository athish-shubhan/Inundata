import numpy as np
import pandas as pd

from src.config import DATA_PROC
from src.geo.grid import read_stack

# Everything here is knowable before the flood. Post-event imagery and the
# reference extent are never predictors (tests/test_ml.py enforces this).
FEATURE_COLS = ["elev_m", "slope_deg", "hand_m", "river_dist_m", "tpi_m", "log_flow_acc",
                "ndvi_pre", "ndwi_pre", "ndbi_pre"]
LABEL_COL = "flooded"
BLOCK_M = 2000


def load_table(path=None) -> pd.DataFrame:
    """One row per grid cell that can be scored: not open water before the event and not cloudy."""
    grid, layers, _ = read_stack(path or DATA_PROC / "features.tif")
    x, y = grid.xy()
    lon, lat = grid.lonlat()
    df = pd.DataFrame({k: v.ravel() for k, v in layers.items()})
    df["row"], df["col"] = np.divmod(np.arange(grid.height * grid.width), grid.width)
    df["x"], df["y"], df["lon"], df["lat"] = x.ravel(), y.ravel(), lon.ravel(), lat.ravel()
    keep = (df.permanent_water == 0) & (df.cloud == 0) & df[FEATURE_COLS].notna().all(axis=1)
    return df[keep].reset_index(drop=True)


def spatial_blocks(df: pd.DataFrame, block_m: float = BLOCK_M) -> np.ndarray:
    """Square blocks in metres. Cells in the same block always land in the same CV fold."""
    bx = ((df.x - df.x.min()) // block_m).astype(int)
    by = ((df.y - df.y.min()) // block_m).astype(int)
    return (by * (bx.max() + 1) + bx).to_numpy()
