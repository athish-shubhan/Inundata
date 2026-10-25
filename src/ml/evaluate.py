"""Where the model is wrong, from out-of-fold scores only (each cell scored by a model that never saw its block)."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import DATA_PROC
from src.geo.grid import read_stack
from src.ml.features import LABEL_COL, load_table

HAND_BANDS_M = [0, 2, 5, 10, 20, np.inf]


def error_report() -> dict:
    df = load_table()
    _, risk, tags = read_stack(DATA_PROC / "risk.tif")
    thr = float(tags["operating_threshold"])
    df["risk_oof"] = risk["risk_oof"][df.row, df.col]
    df["pred"] = (df.risk_oof >= thr).astype(int)
    y = df[LABEL_COL].astype(int)

    df["error_type"] = "correct"
    df.loc[(y == 1) & (df.pred == 0), "error_type"] = "false_negative"
    df.loc[(y == 0) & (df.pred == 1), "error_type"] = "false_positive"
    errors = df[df.error_type != "correct"]
    errors[["lat", "lon", "risk_oof", "hand_m", "elev_m", "river_dist_m", "flood_fraction", "error_type"]].round(4) \
        .to_csv(DATA_PROC / "error_analysis.csv", index=False)

    df["hand_band"] = pd.cut(df.hand_m, HAND_BANDS_M, right=False)
    by_hand = df.groupby("hand_band", observed=True).agg(
        cells=("pred", "size"), observed_rate=(LABEL_COL, "mean"), predicted_rate=("pred", "mean"),
        false_negatives=("error_type", lambda s: int((s == "false_negative").sum())),
        false_positives=("error_type", lambda s: int((s == "false_positive").sum())))
    return {"threshold": thr, "confusion_matrix": confusion_matrix(y, df.pred).tolist(),
            "errors_by_hand_band": by_hand.round(4).reset_index().astype({"hand_band": str}).to_dict("records")}


if __name__ == "__main__":
    print(json.dumps(error_report(), indent=2, default=float))
