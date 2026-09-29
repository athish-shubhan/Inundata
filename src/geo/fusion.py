import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
from src.config import DATA_PROC
from src.geo.terrain import load_grid, slope_deg, to_geotiff
from src.geo.indices import river_distance_m, height_above_drainage, ndwi, ndvi
from src.geo.weather import idw_grid

NDWI_CHANGE_THRESH = 0.04

def neighborhood_features(arr, size=3):
    mean = uniform_filter(np.nan_to_num(arr, nan=np.nanmean(arr)), size=size)
    rng = maximum_filter(arr, size=size) - minimum_filter(arr, size=size)
    return mean.astype("float32"), rng.astype("float32")

def build_features() -> pd.DataFrame:
    lats, lons, elev = load_grid()
    slope = slope_deg(lats, lons, elev)
    river_dist = river_distance_m(lats, lons)
    hand = height_above_drainage(elev, river_dist)
    rain_flood = idw_grid(lats, lons, "flood_event")
    rain_base = idw_grid(lats, lons, "baseline")
    n1 = ndvi("pre_flood_2020-05-11", lats, lons)
    w1 = ndwi("pre_flood_2020-05-11", lats, lons)
    w2 = ndwi("post_flood_2020-08-29", lats, lons)
    ndwi_diff = w2 - w1
    elev_nbr_mean, elev_nbr_rng = neighborhood_features(elev)
    hand_nbr_mean, _ = neighborhood_features(hand)

    lon_g, lat_g = np.meshgrid(lons, lats)
    df = pd.DataFrame({
        "lat": lat_g.ravel(), "lon": lon_g.ravel(), "elev_m": elev.ravel(), "slope_deg": slope.ravel(),
        "hand_m": hand.ravel(), "river_dist_m": river_dist.ravel(),
        "rain_flood_mm": rain_flood.ravel(), "rain_baseline_mm": rain_base.ravel(),
        "rain_anomaly_mm": (rain_flood - rain_base).ravel(),
        "ndvi_pre": n1.ravel(), "ndwi_pre": w1.ravel(), "ndwi_post": w2.ravel(), "ndwi_diff": ndwi_diff.ravel(),
        "elev_nbr_mean": elev_nbr_mean.ravel(), "elev_local_relief": elev_nbr_rng.ravel(),
        "hand_nbr_mean": hand_nbr_mean.ravel(),
    })
    df = df.dropna(subset=["elev_m", "slope_deg", "hand_m", "ndwi_diff"]).reset_index(drop=True)
    df["inundation_signal"] = (df.ndwi_diff > NDWI_CHANGE_THRESH).astype(int)

    out = DATA_PROC / "terrain"
    out.mkdir(parents=True, exist_ok=True)
    to_geotiff(lats, lons, hand, out / "hand.tif")
    to_geotiff(lats, lons, river_dist, out / "river_dist.tif")
    to_geotiff(lats, lons, ndwi_diff, out / "ndwi_diff.tif")
    to_geotiff(lats, lons, rain_flood, out / "rain_flood.tif")
    df.to_csv(DATA_PROC / "features.csv", index=False)
    return df

if __name__ == "__main__":
    df = build_features()
    print(df.shape)
    print(df.inundation_signal.value_counts())
    print(df.describe().T[["mean", "min", "max"]])
