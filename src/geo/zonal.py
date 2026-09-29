import numpy as np

def _nearest_idx(vals, target):
    return int(np.abs(vals - target).argmin())

def sample_grid(lats, lons, arr, lat, lon, window=1):
    i = _nearest_idx(lats, lat)
    j = _nearest_idx(lons, lon)
    i0, i1 = max(0, i - window), min(arr.shape[0], i + window + 1)
    j0, j1 = max(0, j - window), min(arr.shape[1], j + window + 1)
    patch = arr[i0:i1, j0:j1]
    return float(np.nanmean(patch)) if np.isfinite(patch).any() else np.nan

def zonal_sample(gdf, lats, lons, layers: dict, window=1):
    out = gdf.copy()
    for name, arr in layers.items():
        out[name] = [sample_grid(lats, lons, arr, geom.y, geom.x, window) for geom in out.geometry]
    return out
