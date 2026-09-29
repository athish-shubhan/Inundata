import numpy as np
import pandas as pd

FEATURE_COLS = ["elev_m", "slope_deg", "hand_m", "river_dist_m", "rain_flood_mm", "rain_anomaly_mm",
                "ndvi_pre", "ndwi_pre", "elev_nbr_mean", "elev_local_relief", "hand_nbr_mean"]
LABEL_COL = "inundation_signal"

def spatial_groups(df: pd.DataFrame, n_blocks=4) -> np.ndarray:
    lat_bin = pd.cut(df.lat, n_blocks, labels=False)
    lon_bin = pd.cut(df.lon, n_blocks, labels=False)
    return (lat_bin * n_blocks + lon_bin).values

def Xy(df: pd.DataFrame):
    return df[FEATURE_COLS].values, df[LABEL_COL].values
