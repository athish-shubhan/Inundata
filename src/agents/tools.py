import json
import time
import pandas as pd
from math import radians, sin, cos, asin, sqrt
from src.config import AOI_NAME, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, DATA_RAW, DATA_PROC
from src.ml.model import RiskModel

_rm = None

def _model() -> RiskModel:
    global _rm
    if _rm is None:
        _rm = RiskModel()
    return _rm

def haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(radians, [la1, lo1, la2, lo2])
    d = 2 * asin(sqrt(sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2))
    return 6371 * d

def get_region_info() -> dict:
    return {"name": AOI_NAME, "bounds": {"lat_min": LAT_MIN, "lat_max": LAT_MAX, "lon_min": LON_MIN, "lon_max": LON_MAX},
            "context": "Kuma River basin, site of the July 2020 Kyushu flood disaster."}

def assess_region(lat: float, lon: float) -> dict:
    return _model().predict_point(lat, lon)

def list_exposed_assets(lat: float, lon: float, radius_km: float = 2.0) -> dict:
    client = pd.read_csv(DATA_RAW / "client" / "client_portfolio.csv")
    client["dist_km"] = client.apply(lambda r: haversine_km(lat, lon, r.lat, r.lon), axis=1)
    near = client[client.dist_km <= radius_km].copy()
    if near.empty:
        return {"n_assets": 0, "assets": [], "total_value_jpy": 0}
    scored = _model().predict_points(near)
    scored = scored.sort_values("risk_score", ascending=False)
    return {"n_assets": len(scored), "total_value_jpy": int(scored.value_jpy.sum()),
            "assets": scored[["asset_id", "building_type", "value_jpy", "dist_km", "risk_score"]].round(3).to_dict("records")}

def get_model_metrics() -> dict:
    return json.loads((DATA_PROC / "metrics.json").read_text())

TOOL_SPECS = [
    {"name": "get_region_info", "description": "Get the AOI name, bounding box, and disaster context for the covered region.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "assess_region", "description": "Get flood-susceptibility risk score (0-1) and driving features at a lat/lon point.",
     "input_schema": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}}, "required": ["lat", "lon"]}},
    {"name": "list_exposed_assets", "description": "List client property-portfolio assets within radius_km of a point, scored by flood risk, sorted highest-risk first.",
     "input_schema": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}, "radius_km": {"type": "number"}}, "required": ["lat", "lon"]}},
    {"name": "get_model_metrics", "description": "Get the model's cross-validated performance metrics and known caveats.",
     "input_schema": {"type": "object", "properties": {}}},
]

DISPATCH = {"get_region_info": get_region_info, "assess_region": assess_region,
            "list_exposed_assets": list_exposed_assets, "get_model_metrics": get_model_metrics}

def call_tool(name: str, args: dict, log: list) -> dict:
    t0 = time.time()
    if name not in DISPATCH:
        result = {"error": f"unknown tool {name}"}
    else:
        try:
            result = DISPATCH[name](**args)
        except Exception as e:
            result = {"error": str(e)}
    log.append({"tool": name, "args": args, "latency_ms": round((time.time() - t0) * 1000, 1),
                "ok": "error" not in result})
    return result
