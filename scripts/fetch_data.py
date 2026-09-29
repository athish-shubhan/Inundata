import sys, json, time, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests
from src.config import DATA_RAW, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_STEP, FLOOD_EVENT_START, FLOOD_EVENT_END, BASELINE_START, BASELINE_END

def grid_points():
    lats = [round(LAT_MIN + i * GRID_STEP, 5) for i in range(int((LAT_MAX - LAT_MIN) / GRID_STEP) + 1)]
    lons = [round(LON_MIN + i * GRID_STEP, 5) for i in range(int((LON_MAX - LON_MIN) / GRID_STEP) + 1)]
    return lats, lons

def fetch_elevation():
    out = DATA_RAW / "terrain" / "srtm_points.csv"
    if out.exists():
        print("elevation cached:", out); return
    lats, lons = grid_points()
    pts = [(la, lo) for la in lats for lo in lons]
    rows = []
    for i in range(0, len(pts), 100):
        batch = pts[i:i + 100]
        loc = "|".join(f"{la},{lo}" for la, lo in batch)
        r = requests.get("https://api.opentopodata.org/v1/srtm90m", params={"locations": loc}, timeout=30)
        r.raise_for_status()
        res = r.json()["results"]
        for (la, lo), rr in zip(batch, res):
            rows.append((la, lo, rr["elevation"]))
        print(f"elevation {i+len(batch)}/{len(pts)}")
        time.sleep(1.1)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lat", "lon", "elev_m"]); w.writerows(rows)
    print("saved", out)

def fetch_weather():
    stations = {
        "hitoyoshi_center": (32.795, 130.755),
        "kuma_upper": (32.77, 130.70),
        "kuma_lower": (32.83, 130.80),
    }
    for name, (la, lo) in stations.items():
        for label, (s, e) in [("flood_event", (FLOOD_EVENT_START, FLOOD_EVENT_END)), ("baseline", (BASELINE_START, BASELINE_END))]:
            out = DATA_RAW / "weather" / f"{name}_{label}.csv"
            if out.exists():
                continue
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": la, "longitude": lo, "start_date": s, "end_date": e,
                "daily": "precipitation_sum,precipitation_hours", "timezone": "Asia/Tokyo",
            }, timeout=30)
            r.raise_for_status()
            d = r.json()["daily"]
            with open(out, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["date", "precip_mm", "precip_hours"])
                for dt, p, h in zip(d["time"], d["precipitation_sum"], d["precipitation_hours"]):
                    w.writerow([dt, p, h])
            print("saved", out)
            time.sleep(0.5)

def build_river():
    out = DATA_RAW / "gis" / "kuma_river.geojson"
    coords = [
        [130.685, 32.775], [130.705, 32.778], [130.722, 32.783],
        [130.742, 32.790], [130.758, 32.795], [130.770, 32.805],
        [130.782, 32.815], [130.795, 32.828], [130.805, 32.845],
    ]
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"name": "Kuma River (simplified centerline)", "source": "manually digitized for PoC, approximate"},
        "geometry": {"type": "LineString", "coordinates": coords},
    }]}
    out.write_text(json.dumps(fc, indent=2))
    print("saved", out)

def build_infrastructure():
    import random
    random.seed(7)
    out = DATA_RAW / "gis" / "infrastructure.geojson"
    kinds = [("hospital", 3), ("school", 6), ("bridge", 4), ("evacuation_center", 5), ("fire_station", 2)]
    feats = []
    fid = 0
    for kind, n in kinds:
        for _ in range(n):
            la = round(random.uniform(LAT_MIN + 0.01, LAT_MAX - 0.01), 5)
            lo = round(random.uniform(LON_MIN + 0.01, LON_MAX - 0.01), 5)
            fid += 1
            feats.append({"type": "Feature", "properties": {"id": f"infra_{fid}", "kind": kind, "name": f"{kind}_{fid}"},
                           "geometry": {"type": "Point", "coordinates": [lo, la]}})
    fc = {"type": "FeatureCollection", "properties": {"source": "simulated critical-infrastructure register for PoC"}, "features": feats}
    out.write_text(json.dumps(fc, indent=2))
    print("saved", out)

def build_client_portfolio():
    import random
    random.seed(42)
    out = DATA_RAW / "client" / "client_portfolio.csv"
    types = ["residential", "commercial", "industrial", "agricultural"]
    rows = [["asset_id", "lat", "lon", "building_type", "value_jpy", "year_built", "occupants"]]
    for i in range(120):
        la = round(random.uniform(LAT_MIN + 0.005, LAT_MAX - 0.005), 5)
        lo = round(random.uniform(LON_MIN + 0.005, LON_MAX - 0.005), 5)
        bt = random.choice(types)
        val = random.randint(8_000_000, 95_000_000) if bt != "agricultural" else random.randint(500_000, 4_000_000)
        yr = random.randint(1965, 2019)
        occ = random.randint(0, 45) if bt != "agricultural" else 0
        rows.append([f"C{i+1:04d}", la, lo, bt, val, yr, occ])
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("saved", out, "(simulated client property portfolio)")

if __name__ == "__main__":
    fetch_elevation()
    fetch_weather()
    build_river()
    build_infrastructure()
    build_client_portfolio()
