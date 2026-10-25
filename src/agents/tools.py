import json
import time
from functools import cache

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import AOI, AOI_NAME, DATA_PROC, DATA_RAW, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
from src.geo.weather import event_rainfall
from src.ml.model import RiskModel

_rm = None


def _model() -> RiskModel:
    global _rm
    if _rm is None:
        _rm = RiskModel()
    return _rm


def haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    d = 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    return 6371 * d


@cache
def _portfolio() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "client" / "client_portfolio.csv")


@cache
def _infrastructure() -> pd.DataFrame:
    gdf = gpd.read_file(DATA_RAW / "gis" / "infrastructure.geojson")
    return pd.DataFrame(gdf.drop(columns="geometry")).assign(lat=gdf.geometry.y, lon=gdf.geometry.x)


def _within(df: pd.DataFrame, lat: float, lon: float, radius_km: float) -> pd.DataFrame:
    df = df.assign(dist_km=haversine_km(lat, lon, df.lat.to_numpy(), df.lon.to_numpy()))
    return df[df.dist_km <= radius_km]


def get_region_info() -> dict:
    return {"name": AOI_NAME, "context": AOI["context"],
            "bounds": {"lat_min": LAT_MIN, "lat_max": LAT_MAX, "lon_min": LON_MIN, "lon_max": LON_MAX},
            "event_window": AOI["event"], "known_places": sorted(AOI["places"])}


def assess_region(lat: float, lon: float) -> dict:
    return _model().predict_point(lat, lon)


def list_exposed_assets(lat: float, lon: float, radius_km: float = 2.0) -> dict:
    near = _within(_portfolio(), lat, lon, radius_km)
    if near.empty:
        return {"n_assets": 0, "total_value_jpy": 0, "n_high_risk": 0, "high_risk_value_jpy": 0, "assets": []}
    scored = _model().predict_points(near).sort_values("risk_score", ascending=False, na_position="last")
    high = scored[scored.risk_band == "HIGH"]
    cols = ["asset_id", "building_type", "value_jpy", "occupants", "dist_km", "risk_score", "risk_band", "flooded_2020"]
    return {"n_assets": len(scored), "total_value_jpy": int(scored.value_jpy.sum()),
            "n_high_risk": len(high), "high_risk_value_jpy": int(high.value_jpy.sum()),
            "assets": json.loads(scored[cols].round(3).to_json(orient="records"))}


def list_exposed_infrastructure(lat: float, lon: float, radius_km: float = 2.0) -> dict:
    near = _within(_infrastructure(), lat, lon, radius_km)
    if near.empty:
        return {"n_sites": 0, "n_high_risk": 0, "sites": []}
    scored = _model().predict_points(near).sort_values("risk_score", ascending=False, na_position="last")
    cols = ["id", "kind", "name", "dist_km", "risk_score", "risk_band", "flooded_2020"]
    return {"n_sites": len(scored), "n_high_risk": int((scored.risk_band == "HIGH").sum()),
            "sites": json.loads(scored[cols].round(3).to_json(orient="records", force_ascii=False))}


def get_event_rainfall() -> dict:
    return event_rainfall()


def get_model_metrics() -> dict:
    m = json.loads((DATA_PROC / "metrics.json").read_text())
    best = m["selected_model"]
    summary = lambda cv: {k.removesuffix("_mean"): round(v, 3) for k, v in cv.items() if k.endswith("_mean")}
    return {"selected_model": best, "operating_threshold": round(m["operating_threshold"], 3),
            "spatial_cv": summary(m["cv"][best]["spatial_cv"]), "random_cv": summary(m["cv"][best]["random_cv"]),
            "baseline_hand_threshold_spatial_cv": summary(m["cv"]["hand_threshold"]["spatial_cv"]),
            "prevalence": round(m["data"]["prevalence"], 4), "n_cells": m["data"]["n_cells"],
            "top_features": list(m["feature_importance"])[:4], "caveat": m["caveat"]}


_POINT = {"lat": {"type": "number"}, "lon": {"type": "number"}}
_RADIUS = {"radius_km": {"type": "number", "description": "search radius, default 2"}}

TOOL_SPECS = [
    {"name": "get_region_info", "description": "Get the AOI name, bounding box, flood-event context and the named places the system knows.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "assess_region", "description": "Calibrated flood-susceptibility score (0-1), risk band and terrain/land-cover drivers for the 30 m grid cell at a lat/lon, plus whether that cell was inside the observed 2020 flood extent.",
     "input_schema": {"type": "object", "properties": _POINT, "required": ["lat", "lon"]}},
    {"name": "list_exposed_assets", "description": "Client property-portfolio assets within radius_km of a point, scored by flood risk, highest risk first, with total and high-risk exposure in JPY.",
     "input_schema": {"type": "object", "properties": _POINT | _RADIUS, "required": ["lat", "lon"]}},
    {"name": "list_exposed_infrastructure", "description": "Critical infrastructure (hospitals, clinics, schools, kindergartens, fire stations, police, town hall; from OpenStreetMap) within radius_km of a point, scored by flood risk.",
     "input_schema": {"type": "object", "properties": _POINT | _RADIUS, "required": ["lat", "lon"]}},
    {"name": "get_event_rainfall", "description": "Rainfall totals for the July 2020 event window and a baseline window (Open-Meteo reanalysis). Context only; not a model input.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_model_metrics", "description": "The model's spatially cross-validated performance, the random-CV figure for comparison, a terrain-only baseline, and known caveats.",
     "input_schema": {"type": "object", "properties": {}}},
]

DISPATCH = {"get_region_info": get_region_info, "assess_region": assess_region,
            "list_exposed_assets": list_exposed_assets, "list_exposed_infrastructure": list_exposed_infrastructure,
            "get_event_rainfall": get_event_rainfall, "get_model_metrics": get_model_metrics}


def call_tool(name: str, args: dict, log: list) -> dict:
    t0 = time.perf_counter()
    if name not in DISPATCH:
        result = {"error": f"unknown tool {name}"}
    else:
        try:
            result = DISPATCH[name](**args)
        except Exception as e:
            result = {"error": str(e)}
    log.append({"tool": name, "args": args, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "ok": "error" not in result})
    return result
