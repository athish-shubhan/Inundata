import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from src.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, DATA_RAW

SCENES = {
    "pre_flood_2020-05-11": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/52/S/FB/2020/5/S2A_52SFB_20200511_2_L2A",
    "post_flood_2020-08-29": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/52/S/FB/2020/8/S2A_52SFB_20200829_0_L2A",
}
BANDS = ["B03", "B04", "B08", "B11", "SCL"]

def clip_band(url, out_path):
    if out_path.exists():
        return
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(url) as src:
            b = transform_bounds("EPSG:4326", src.crs, LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
            win = from_bounds(*b, transform=src.transform)
            arr = src.read(1, window=win)
            prof = src.profile.copy()
            prof.update(height=arr.shape[0], width=arr.shape[1], transform=src.window_transform(win))
            with rasterio.open(out_path, "w", **prof) as dst:
                dst.write(arr, 1)
    print("saved", out_path, arr.shape)

if __name__ == "__main__":
    out_dir = DATA_RAW / "satellite"
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, base in SCENES.items():
        for band in BANDS:
            clip_band(f"{base}/{band}.tif", out_dir / f"{label}_{band}.tif")
