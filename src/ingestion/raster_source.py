import rasterio
import numpy as np
from .base import IngestReport

def load_raster(path):
    rep = IngestReport(source=str(path))
    try:
        ds = rasterio.open(path)
    except Exception as e:
        rep.errors.append(f"failed to read raster: {e}")
        return None, rep

    arr = ds.read(1)
    nod = ds.nodata
    n_nod = int((arr == nod).sum()) if nod is not None else int(np.isnan(arr).sum())
    if n_nod:
        rep.warnings.append(f"{n_nod} nodata pixels")
    if ds.crs is None:
        rep.errors.append("raster has no CRS")
    rep.n_rows = arr.size
    rep.meta.update({"crs": str(ds.crs), "shape": arr.shape, "bounds": list(ds.bounds),
                      "resolution": ds.res, "dtype": str(arr.dtype)})
    return ds, rep
