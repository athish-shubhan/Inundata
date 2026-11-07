"""Hindcast of the simulated client portfolio against the 2020 GSI flood extent.

    python scripts/hindcast.py --out results/portfolio_hindcast.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from src.config import DATA_RAW
    from src.ml.model import RiskModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    df = pd.read_csv(DATA_RAW / "client" / "client_portfolio.csv")
    sc = RiskModel().predict_points(df)
    high = sc["risk_band"] == "HIGH"
    flooded = sc["flooded_2020"].fillna(False).astype(bool)
    out = {"n_assets": int(len(sc)), "n_high": int(high.sum()),
           "n_flooded_2020": int(flooded.sum()),
           "flooded_and_high": int((high & flooded).sum()),
           "high_but_dry": int((high & ~flooded).sum())}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
