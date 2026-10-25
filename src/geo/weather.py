"""Event rainfall for the report narrative.

Rainfall is deliberately not a model feature: Open-Meteo's archive is ERA5-based
reanalysis on a roughly 10 km grid, so across a 15 km AOI it is close to
constant and only lets a model memorise location. It is reported as event
context instead.
"""
import pandas as pd

from src.config import AOI, DATA_RAW


def event_rainfall() -> dict:
    out = {}
    for name in AOI["weather_points"]:
        event = pd.read_csv(DATA_RAW / "weather" / f"{name}_event.csv")
        base = pd.read_csv(DATA_RAW / "weather" / f"{name}_baseline.csv")
        out[name] = {"event_total_mm": round(float(event.precip_mm.sum()), 1),
                     "event_max_day_mm": round(float(event.precip_mm.max()), 1),
                     "event_max_day": str(event.loc[event.precip_mm.idxmax(), "date"]),
                     "baseline_total_mm": round(float(base.precip_mm.sum()), 1)}
    return {"window": f"{AOI['event']['start']} to {AOI['event']['end']}",
            "baseline_window": f"{AOI['baseline']['start']} to {AOI['baseline']['end']}",
            "points": out}
