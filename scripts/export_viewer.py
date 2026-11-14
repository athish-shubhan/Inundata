"""Export the study area for the 3D viewer in docs/viewer/.

Reads the pipeline's own outputs (features.tif, risk.tif, metrics.json) and the
committed vector data, and writes two files the browser can load directly:

  docs/viewer/data/terrain.bin   per-cell layers, row-major, north row first
  docs/viewer/data/scene.json    grid description, river, assets, facilities

Nothing is modelled here. Every number in the viewer is a value the pipeline
already produced; this script only repacks it.

terrain.bin layout for a grid of n = width * height cells:
  elev   uint16[n]  elevation in decimetres above ELEV_OFFSET_M
  hand   uint8[n]   height above nearest drainage in HAND_STEP_M steps (capped)
  risk   uint8[n]   model risk scaled to 0..254; 255 = no prediction
  flags  uint8[n]   bit 0 observed flood (GSI), bit 1 permanent water
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AOI, AOI_NAME, CRS, DATA_PROC, DATA_RAW, ROOT

OUT_DIR = ROOT / "docs" / "viewer" / "data"

HAND_STEP_M = 0.25
RISK_NODATA = 255
FLAG_FLOODED = 1
FLAG_PERMANENT_WATER = 2


def pack_layers(elev_m, hand_m, risk, flooded, permanent_water):
    """Quantise the float layers. Returns (bytes, elev_offset_m)."""
    elev_offset = float(np.floor(np.nanmin(elev_m)))
    elev = np.clip(np.round((elev_m - elev_offset) * 10), 0, 65535).astype("<u2")
    hand = np.clip(np.round(hand_m / HAND_STEP_M), 0, 255).astype(np.uint8)
    risk_q = np.full(risk.shape, RISK_NODATA, dtype=np.uint8)
    ok = ~np.isnan(risk)
    risk_q[ok] = np.clip(np.round(risk[ok] * 254), 0, 254).astype(np.uint8)
    flags = ((flooded > 0.5) * FLAG_FLOODED + (permanent_water > 0.5) * FLAG_PERMANENT_WATER).astype(np.uint8)
    return elev.tobytes() + hand.tobytes() + risk_q.tobytes() + flags.tobytes(), elev_offset


def confusion(risk, flooded, threshold):
    """Cell counts of model-vs-observed at the operating threshold."""
    ok = ~np.isnan(risk)
    pred, obs = risk[ok] >= threshold, flooded[ok] > 0.5
    return {
        "hit": int((pred & obs).sum()),
        "miss": int((~pred & obs).sum()),
        "false_alarm": int((pred & ~obs).sum()),
        "correct_dry": int((~pred & ~obs).sum()),
    }


def to_grid(lons, lats, transform, crs):
    """Lon/lat -> fractional (col, row) on the raster grid."""
    if transform.b or transform.d:
        raise ValueError("the grid is expected to be north-up (no rotation)")
    xs, ys = warp_transform(CRS, crs, list(lons), list(lats))
    return [((x - transform.c) / transform.a, (y - transform.f) / transform.e) for x, y in zip(xs, ys)]


def build_scene():
    with rasterio.open(DATA_PROC / "features.tif") as src:
        names = list(src.descriptions)
        band = {n: src.read(names.index(n) + 1).astype(np.float64) for n in
                ("elev_m", "hand_m", "flooded", "permanent_water")}
        transform, crs, width, height = src.transform, src.crs, src.width, src.height
    with rasterio.open(DATA_PROC / "risk.tif") as src:
        if (src.width, src.height) != (width, height) or src.transform != transform:
            raise ValueError("risk.tif and features.tif are not on the same grid")
        risk = src.read(1).astype(np.float64)
    metrics = json.loads((DATA_PROC / "metrics.json").read_text())
    threshold = float(metrics["operating_threshold"])

    blob, elev_offset = pack_layers(band["elev_m"], band["hand_m"], risk,
                                    band["flooded"], band["permanent_water"])

    def inside(col, row):
        return 0 <= col < width and 0 <= row < height

    def cell(col, row):
        return int(row), int(col)

    river = []
    for feat in json.loads((DATA_RAW / "gis" / "river.geojson").read_text())["features"]:
        geom = feat["geometry"]
        lines = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
        for line in lines:
            pts = to_grid([p[0] for p in line], [p[1] for p in line], transform, crs)
            pts = [[round(c, 2), round(r, 2)] for c, r in pts if inside(c, r)]
            if len(pts) >= 2:
                river.append(pts)

    assets = []
    with open(DATA_RAW / "client" / "client_portfolio.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for row, (c, r) in zip(rows, to_grid([float(x["lon"]) for x in rows], [float(x["lat"]) for x in rows],
                                         transform, crs)):
        if not inside(c, r):
            continue
        value = risk[cell(c, r)]
        assets.append({
            "id": row["asset_id"], "type": row["building_type"], "value_jpy": int(float(row["value_jpy"])),
            "col": round(c, 2), "row": round(r, 2),
            "risk": None if np.isnan(value) else round(float(value), 3),
            "flooded": bool(band["flooded"][cell(c, r)] > 0.5),
        })

    facilities = []
    feats = json.loads((DATA_RAW / "gis" / "infrastructure.geojson").read_text())["features"]
    feats = [f for f in feats if f["geometry"]["type"] == "Point"]
    for feat, (c, r) in zip(feats, to_grid([f["geometry"]["coordinates"][0] for f in feats],
                                           [f["geometry"]["coordinates"][1] for f in feats], transform, crs)):
        if not inside(c, r):
            continue
        value = risk[cell(c, r)]
        facilities.append({
            "kind": feat["properties"].get("kind", ""), "name": feat["properties"].get("name", ""),
            "col": round(c, 2), "row": round(r, 2),
            "risk": None if np.isnan(value) else round(float(value), 3),
            "flooded": bool(band["flooded"][cell(c, r)] > 0.5),
        })

    scene = {
        "name": AOI_NAME,
        "event": AOI["event"],
        "width": width, "height": height, "cell_m": abs(transform.a),
        "elev_offset_m": elev_offset,
        "elev_min_m": round(float(np.nanmin(band["elev_m"])), 1),
        "elev_max_m": round(float(np.nanmax(band["elev_m"])), 1),
        "hand_step_m": HAND_STEP_M,
        "risk_nodata": RISK_NODATA,
        "model": metrics["selected_model"],
        "threshold": round(threshold, 4),
        "confusion": confusion(risk, band["flooded"], threshold),
        "observed_flood_cells": int((band["flooded"] > 0.5).sum()),
        "river": river, "assets": assets, "facilities": facilities,
    }
    return blob, scene


def main():
    blob, scene = build_scene()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "terrain.bin").write_bytes(blob)
    (OUT_DIR / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, separators=(",", ":")))
    print(f"terrain.bin  {len(blob) / 1024:.0f} KB   {scene['width']} x {scene['height']} cells")
    print(f"scene.json   {len(scene['assets'])} assets, {len(scene['facilities'])} facilities, "
          f"{len(scene['river'])} river segments")
    print(f"model {scene['model']} at threshold {scene['threshold']}: {scene['confusion']}")


if __name__ == "__main__":
    main()
