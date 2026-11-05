import numpy as np
import rasterio

from .base import IngestReport, IngestResult


def load_raster(path) -> IngestResult:
    """Validate a single-band raster and report its metadata.

    Only band 1 is inspected. The dataset is closed before returning: ingestion
    is a validation step, and callers (the API, the dashboard) only use the report.
    """
    rep = IngestReport(source=str(path))
    try:
        ds = rasterio.open(path)
    except Exception as e:
        rep.errors.append(f"failed to read raster: {e}")
        return IngestResult(None, rep)

    with ds:
        arr = ds.read(1)
        nod = ds.nodata
        is_float = np.issubdtype(arr.dtype, np.floating)
        if nod is not None and not np.isnan(nod):
            n_nod = int((arr == nod).sum())
        else:  # nodata unset or NaN: NaN pixels are the nodata ones (NaN == NaN is always False)
            n_nod = int(np.isnan(arr).sum()) if is_float else 0
        if n_nod:
            rep.warnings.append(f"{n_nod} nodata pixels")
        if ds.crs is None:
            rep.errors.append("raster has no CRS")
        rep.n_rows = arr.size
        rep.meta.update({"crs": str(ds.crs), "shape": list(arr.shape), "bounds": list(ds.bounds),
                         "resolution": list(ds.res), "dtype": str(arr.dtype)})
    return IngestResult(None, rep)
