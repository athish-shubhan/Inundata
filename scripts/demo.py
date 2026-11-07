"""Command-line demo: flood risk for a named place or coordinate, with explained drivers.

    python scripts/demo.py                       # downtown Hitoyoshi
    python scripts/demo.py --place nishise
    python scripts/demo.py --lat 32.2158 --lon 130.7535 --radius-km 1
    python scripts/demo.py --list-places

Uses only the committed rasters; no network, no API key.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from src.agents import tools
    from src.config import AOI

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", default="downtown")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--radius-km", type=float, default=2.0)
    ap.add_argument("--list-places", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the raw tool output")
    a = ap.parse_args()

    places = AOI["places"]
    if a.list_places:
        print("\n".join(sorted(places)))
        return
    if a.lat is None or a.lon is None:
        if a.place.lower() not in places:
            sys.exit(f"unknown place {a.place!r}; try --list-places")
        a.lat, a.lon = places[a.place.lower()]

    risk = tools.assess_region(a.lat, a.lon)
    assets = tools.list_exposed_assets(a.lat, a.lon, a.radius_km)
    infra = tools.list_exposed_infrastructure(a.lat, a.lon, a.radius_km)
    if a.json:
        print(json.dumps({"risk": risk, "assets": assets, "infrastructure": infra}, indent=2, default=str))
        return
    print(f"Location: {a.lat:.4f}, {a.lon:.4f}")
    if risk.get("status") != "scored":
        print(json.dumps(risk, indent=2, ensure_ascii=False, default=str))
        return
    print(f"Flood risk: {risk['risk_band']} (calibrated score {risk['risk_score']:.2f})")
    obs = risk.get("observed_2020") or {}
    if obs:
        print(f"Flooded in the 2020-07-04 GSI extent: {'yes' if obs.get('flooded') else 'no'} "
              f"(flooded fraction of cell {obs.get('flood_fraction')})")
    d, pct = risk["drivers"], risk.get("driver_percentiles", {})
    units = {"elev_m": "m", "slope_deg": "deg", "hand_m": "m", "river_dist_m": "m"}
    print("Drivers:")
    for k, v in d.items():
        extra = f"  (lower than {100 - pct[k]:.0f}% of scored cells)" if k in pct else ""
        print(f"  {k:14s} {v:9.3f} {units.get(k, '')}{extra}")
    print(f"\nWithin {a.radius_km} km:")
    for name, res in (("client assets", assets), ("critical infrastructure", infra)):
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in res.items()
                                          if not isinstance(v, (list, dict))))


if __name__ == "__main__":
    main()
