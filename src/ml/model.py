"""Serving-side view of the trained model.

Training writes calibrated scores for every grid cell to risk.tif, so serving is
a raster lookup: no sklearn model is loaded and scoring 1 point or 10k points
costs the same coordinate transform plus an array index.
"""
import json
import warnings

import numpy as np
import pandas as pd

from src.config import DATA_PROC
from src.geo.grid import read_stack
from src.ml.features import FEATURE_COLS

MODERATE_MIN = 0.1  # roughly twice the event-wide flooded share; HIGH starts at the trained operating threshold


class RiskModel:
    def __init__(self, proc_dir=DATA_PROC):
        self.grid, self.layers, _ = read_stack(proc_dir / "features.tif")
        _, risk, tags = read_stack(proc_dir / "risk.tif")
        self.risk = risk["risk"]
        self.risk_oof = risk["risk_oof"]
        self.threshold = float(tags["operating_threshold"])
        self.name = json.loads((proc_dir / "metrics.json").read_text())["selected_model"]
        self.lon, self.lat = self.grid.lonlat()

    def band(self, score) -> str:
        if score is None or np.isnan(score):
            return "UNSCORED"
        return "HIGH" if score >= self.threshold else "MODERATE" if score >= MODERATE_MIN else "LOW"

    def _status(self, r, c) -> str:
        if self.layers["permanent_water"][r, c] == 1:
            return "open_water"
        if self.layers["cloud"][r, c] == 1:
            return "cloud"
        return "scored" if np.isfinite(self.risk[r, c]) else "no_data"

    def predict_point(self, lat: float, lon: float) -> dict:
        r, c, inside = self.grid.cell_of(lat, lon)
        if not inside[0]:
            raise ValueError(f"({lat}, {lon}) is outside the analysis area")
        r, c = int(r[0]), int(c[0])
        score = float(self.risk[r, c])
        score = score if np.isfinite(score) else None
        return {"lat": lat, "lon": lon, "risk_score": score, "risk_band": self.band(score),
                "status": self._status(r, c),
                "grid_cell": {"row": r, "col": c, "lat": float(self.lat[r, c]), "lon": float(self.lon[r, c]),
                              "size_m": self.grid.res},
                "drivers": {k: round(float(self.layers[k][r, c]), 3) for k in FEATURE_COLS},
                "observed_2020": {"flood_fraction": round(float(self.layers["flood_fraction"][r, c]), 3),
                                  "flooded": bool(self.layers["flooded"][r, c] == 1)}}

    def predict_points(self, df: pd.DataFrame, lat_col="lat", lon_col="lon") -> pd.DataFrame:
        """Vectorised point scoring. Points outside the grid or on unscored cells get NaN."""
        out = df.copy()
        r, c, inside = self.grid.cell_of(out[lat_col].to_numpy(), out[lon_col].to_numpy())
        r, c = np.where(inside, r, 0), np.where(inside, c, 0)
        out["risk_score"] = np.where(inside, self.risk[r, c], np.nan)
        out["risk_band"] = [self.band(s) for s in out.risk_score]
        out["flooded_2020"] = np.where(inside, self.layers["flooded"][r, c] == 1, False)
        return out

    def coarse_grid(self, step: int = 10) -> pd.DataFrame:
        """Block means over step x step cells, for maps and the /grid endpoint (one row per block)."""
        h, w = (self.grid.height // step) * step, (self.grid.width // step) * step

        def pool(a):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks (open water) stay NaN
                return np.nanmean(a[:h, :w].reshape(h // step, step, w // step, step), axis=(1, 3))

        df = pd.DataFrame({"lat": pool(self.lat).ravel(), "lon": pool(self.lon).ravel(),
                           "risk_score": pool(self.risk).ravel(),
                           "flood_fraction_2020": pool(self.layers["flood_fraction"]).ravel(),
                           "hand_m": pool(self.layers["hand_m"]).ravel(),
                           "elev_m": pool(self.layers["elev_m"]).ravel()})
        return df.dropna(subset=["risk_score"]).reset_index(drop=True)
