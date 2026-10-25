"""In-process API latency benchmark.

Runs each endpoint N times through FastAPI's TestClient (no network hop) and
reports p50/p95 in milliseconds. Cold start is measured separately because the
first request used to pay for lazy model loading.

    python scripts/bench_api.py --n 50 --out benchmarks/api_latency.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    t0 = time.perf_counter()
    from fastapi.testclient import TestClient
    from src.api.main import app
    import_s = time.perf_counter() - t0

    calls = {
        "GET /predict": lambda c: c.get("/predict", params={"lat": args.lat, "lon": args.lon}),
        "GET /assets/exposed": lambda c: c.get("/assets/exposed", params={"lat": args.lat, "lon": args.lon, "radius_km": 2}),
        "GET /grid": lambda c: c.get("/grid"),
        "POST /agent/analyze": lambda c: c.post("/agent/analyze", json={"query": f"flood risk at {args.lat}, {args.lon} within 2km"}),
    }
    results = {"import_s": round(import_s, 2)}
    with TestClient(app) as c:
        t = time.perf_counter()
        r = c.get("/predict", params={"lat": args.lat, "lon": args.lon})
        results["cold_first_request_ms"] = round((time.perf_counter() - t) * 1000, 1)
        assert r.status_code == 200, r.text
        for name, fn in calls.items():
            lat_ms = []
            for _ in range(args.n):
                t = time.perf_counter()
                r = fn(c)
                lat_ms.append((time.perf_counter() - t) * 1000)
                assert r.status_code == 200, (name, r.status_code, r.text[:200])
            results[name] = {"p50_ms": round(statistics.median(lat_ms), 1), "p95_ms": round(pct(lat_ms, 95), 1),
                             "bytes": len(r.content)}
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
