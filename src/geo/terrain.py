import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from src.config import DATA_RAW, DATA_PROC, GRID_STEP, CRS

def load_grid():
    df = pd.read_csv(DATA_RAW / "terrain" / "srtm_points.csv")
    lats = np.sort(df.lat.unique())[::-1]
    lons = np.sort(df.lon.unique())
    elev = df.pivot(index="lat", columns="lon", values="elev_m").reindex(index=lats, columns=lons).values
    elev = np.where(elev < -100, np.nan, elev)
    return lats, lons, elev.astype("float32")

def to_geotiff(lats, lons, arr, path, dtype="float32"):
    tf = from_origin(lons[0] - GRID_STEP / 2, lats[0] + GRID_STEP / 2, GRID_STEP, GRID_STEP)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                        count=1, dtype=dtype, crs=CRS, transform=tf, nodata=np.nan) as dst:
        dst.write(arr.astype(dtype), 1)

def slope_deg(lats, lons, elev):
    dy = GRID_STEP * 111_320
    dx = GRID_STEP * 111_320 * np.cos(np.radians(lats.mean()))
    gy, gx = np.gradient(elev, dy, dx)
    return np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))

def build_terrain_layers():
    lats, lons, elev = load_grid()
    slope = slope_deg(lats, lons, elev)
    out = DATA_PROC / "terrain"
    out.mkdir(parents=True, exist_ok=True)
    to_geotiff(lats, lons, elev, out / "dem.tif")
    to_geotiff(lats, lons, slope, out / "slope.tif")
    return lats, lons, elev, slope

if __name__ == "__main__":
    lats, lons, elev, slope = build_terrain_layers()
    print("grid", elev.shape, "elev range", np.nanmin(elev), np.nanmax(elev), "slope range", np.nanmin(slope), np.nanmax(slope))
