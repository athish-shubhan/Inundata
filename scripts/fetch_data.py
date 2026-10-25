"""Download every raw input for the configured AOI into data/<aoi>/raw/.

Each step is idempotent: existing files are kept, so a re-run only fetches what
is missing. Public endpoints (Overpass, Open-Meteo) are rate limited and flaky,
so all HTTP goes through a session with retry and backoff.

    python scripts/fetch_data.py            # everything
    python scripts/fetch_data.py dem river  # selected steps
"""
import csv
import io
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import AOI, DATA_RAW, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

OVERPASS = "https://overpass-api.de/api/interpreter"
BBOX_OSM = f"{LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}"
DEM_BUFFER_DEG = 0.03  # extra margin so flow routing is not distorted at the AOI edge
S2_BANDS = ["B03", "B04", "B08", "B11", "SCL"]
INFRA_TAGS = {"hospital", "clinic", "school", "kindergarten", "fire_station", "police", "townhall"}


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "inundata-fetch/1.0"
    return s


HTTP = session()


def overpass(query: str) -> list[dict]:
    r = HTTP.post(OVERPASS, data={"data": f"[out:json][timeout:170];{query}"}, timeout=200)
    r.raise_for_status()
    return r.json()["elements"]


def save_window(url, bounds_4326, out_path, env=None):
    """Windowed read of a cloud-optimized GeoTIFF: only the bytes for the AOI are downloaded."""
    if out_path.exists():
        return
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", **(env or {})):
        with rasterio.open(url) as src:
            b = transform_bounds("EPSG:4326", src.crs, *bounds_4326)
            win = window_from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win)
            prof = src.profile.copy()
            prof.update(driver="GTiff", height=arr.shape[0], width=arr.shape[1], transform=src.window_transform(win),
                        compress="deflate", tiled=False, blockxsize=None, blockysize=None)
            with rasterio.open(out_path, "w", **prof) as dst:
                dst.write(arr, 1)
    print("saved", out_path, arr.shape)


def fetch_dem():
    out = DATA_RAW / "terrain" / "copernicus_dem30.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    b = DEM_BUFFER_DEG
    save_window(AOI["dem"], (LON_MIN - b, LAT_MIN - b, LON_MAX + b, LAT_MAX + b), out)


def fetch_sentinel2():
    out_dir = DATA_RAW / "satellite"
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, base in AOI["sentinel2"].items():
        scene = base.rsplit("/", 1)[1]
        for band in S2_BANDS:
            save_window(f"{base}/{band}.tif", (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX), out_dir / f"{label}_{band}.tif")
        (out_dir / f"{label}_scene.txt").write_text(scene + "\n")


def fetch_weather():
    out_dir = DATA_RAW / "weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (lat, lon) in AOI["weather_points"].items():
        for label in ("event", "baseline"):
            out = out_dir / f"{name}_{label}.csv"
            if out.exists():
                continue
            r = HTTP.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": lat, "longitude": lon, "start_date": AOI[label]["start"], "end_date": AOI[label]["end"],
                "daily": "precipitation_sum", "timezone": "Asia/Tokyo"}, timeout=30)
            r.raise_for_status()
            d = r.json()["daily"]
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "precip_mm"])
                w.writerows(zip(d["time"], d["precipitation_sum"]))
            print("saved", out)


def fetch_river():
    out = DATA_RAW / "gis" / "river.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    ways = overpass(f'way["waterway"="river"]["name"="{AOI["river_name"]}"]({BBOX_OSM});out geom;')
    feats = [{"type": "Feature", "properties": {"osm_id": w["id"], "name": w["tags"].get("name")},
              "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"]] for p in w["geometry"]]}}
             for w in ways]
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print("saved", out, len(feats), "ways")


def fetch_infrastructure():
    out = DATA_RAW / "gis" / "infrastructure.geojson"
    if out.exists():
        return
    regex = "|".join(sorted(INFRA_TAGS))
    els = overpass(f'nwr["amenity"~"^({regex})$"]({BBOX_OSM});out center tags;')
    feats = []
    for e in els:
        c = e.get("center", e)
        tags = e["tags"]
        feats.append({"type": "Feature",
                      "properties": {"id": f"osm_{e['type']}_{e['id']}", "kind": tags["amenity"],
                                     "name": tags.get("name:en") or tags.get("name") or tags["amenity"]},
                      "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]}})
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False))
    print("saved", out, len(feats), "sites")


def build_client_portfolio(n=120, seed=42):
    """Simulated insurer portfolio placed on real OSM building footprints.

    Locations are real buildings, so exposure analysis reflects where people
    actually live; value, year built and occupants are synthetic.
    """
    out = DATA_RAW / "client" / "client_portfolio.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    els = overpass(f'way["building"]({BBOX_OSM});out center tags;')
    rng = random.Random(seed)
    picks = rng.sample(sorted(els, key=lambda e: e["id"]), n)
    kind_map = {"house": "residential", "detached": "residential", "apartments": "residential", "residential": "residential",
                "retail": "commercial", "commercial": "commercial", "office": "commercial",
                "industrial": "industrial", "warehouse": "industrial", "farm": "agricultural", "barn": "agricultural"}
    rows = [["asset_id", "lat", "lon", "building_type", "value_jpy", "year_built", "occupants"]]
    for i, e in enumerate(picks):
        bt = kind_map.get(e["tags"].get("building"), rng.choice(["residential", "residential", "commercial", "industrial", "agricultural"]))
        val = rng.randint(500_000, 4_000_000) if bt == "agricultural" else rng.randint(8_000_000, 95_000_000)
        occ = 0 if bt == "agricultural" else rng.randint(1, 45)
        rows.append([f"C{i + 1:04d}", round(e["center"]["lat"], 6), round(e["center"]["lon"], 6), bt, val,
                     rng.randint(1965, 2019), occ])
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("saved", out, f"({n} simulated assets on real OSM buildings, sampled from {len(els)})")


def _tile_xy(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def _tile_bounds_3857(x, y, z):
    size = 2 * math.pi * 6378137 / 2 ** z
    origin = math.pi * 6378137
    return x * size - origin, origin - (y + 1) * size, (x + 1) * size - origin, origin - y * size


def fetch_flood_reference():
    """Mosaic the GSI inundation-estimate tiles into a georeferenced 0/1 mask (EPSG:3857).

    Tiles that return 404 contain no inundation; the layer only publishes tiles
    with content.
    """
    out = DATA_RAW / "reference" / "gsi_inundation.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    ref = AOI["flood_reference"]
    z = ref["zoom"]
    x0, y0 = _tile_xy(LAT_MAX, LON_MIN, z)
    x1, y1 = _tile_xy(LAT_MIN, LON_MAX, z)
    mask = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), dtype="uint8")
    hits = 0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            r = HTTP.get(ref["tiles"].format(z=z, x=x, y=y), timeout=30)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            alpha = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))[..., 3]
            mask[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = alpha > 0
            hits += 1
            time.sleep(0.05)
    left, bottom, _, _ = _tile_bounds_3857(x0, y1, z)
    _, _, right, top = _tile_bounds_3857(x1, y0, z)
    tf = from_bounds(left, bottom, right, top, mask.shape[1], mask.shape[0])
    with rasterio.open(out, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1], count=1, dtype="uint8",
                       crs="EPSG:3857", transform=tf, compress="deflate") as dst:
        dst.write(mask, 1)
        dst.update_tags(source=ref["source"], tiles=ref["tiles"], zoom=z)
    print("saved", out, mask.shape, f"{hits} tiles with inundation, {mask.mean():.3%} of pixels flooded")


STEPS = {"dem": fetch_dem, "sentinel2": fetch_sentinel2, "weather": fetch_weather, "river": fetch_river,
         "infrastructure": fetch_infrastructure, "portfolio": build_client_portfolio, "reference": fetch_flood_reference}

if __name__ == "__main__":
    for step in sys.argv[1:] or STEPS:
        STEPS[step]()
