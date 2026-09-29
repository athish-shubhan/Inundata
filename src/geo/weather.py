import numpy as np
import pandas as pd
from src.config import DATA_RAW

STATIONS = {"hitoyoshi_center": (32.795, 130.755), "kuma_upper": (32.77, 130.70), "kuma_lower": (32.83, 130.80)}

def station_totals(label="flood_event"):
    tot = {}
    for name in STATIONS:
        df = pd.read_csv(DATA_RAW / "weather" / f"{name}_{label}.csv")
        tot[name] = df.precip_mm.sum()
    return tot

def idw_grid(lats, lons, label="flood_event", power=2):
    tot = station_totals(label)
    lon_g, lat_g = np.meshgrid(lons, lats)
    out = np.zeros_like(lat_g, dtype="float32")
    wsum = np.zeros_like(lat_g, dtype="float32")
    for name, (sla, slo) in STATIONS.items():
        d = np.sqrt((lat_g - sla) ** 2 + (lon_g - slo) ** 2) * 111_320
        d = np.where(d < 1, 1, d)
        w = 1 / d ** power
        out += w * tot[name]
        wsum += w
    return (out / wsum).astype("float32")
