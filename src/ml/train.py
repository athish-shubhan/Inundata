"""Train and evaluate flood-susceptibility models against the GSI flood extent.

Protocol, per outer fold of spatial CV:
  1. the decision threshold is tuned on an inner split of the training blocks only;
  2. the model is refit on all training blocks and scored on the held-out blocks.
So no number below has seen its test cells, including the threshold.

The same protocol is repeated with a random (non-spatial) KFold to show how much
ordinary CV overstates skill on autocorrelated raster data.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.config import DATA_PROC
from src.geo.grid import read_stack, write_stack
from src.ml.features import BLOCK_M, FEATURE_COLS, LABEL_COL, load_table, spatial_blocks

N_FOLDS = 5
SEED = 0
CAVEAT = ("Label is the GSI rapid inundation estimate for 2020-07-04, itself a model-assisted map rather than a "
          "surveyed extent, and the model is trained and evaluated on this single event. Scores are calibrated on "
          "out-of-fold predictions for this event; treat them as relative susceptibility elsewhere or for other storms.")
# Neighbouring 30 m cells are ~0.9 correlated, so 150k rows carry far less
# information than their count suggests. Each fit sees a random subsample;
# every held-out cell is still scored.
MAX_TRAIN_ROWS = 40_000


class HandThreshold(BaseEstimator, ClassifierMixin):
    """Baseline: "flooded if HAND is below t", the standard terrain-only flood map."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = 1.0 / (1.0 + X[:, FEATURE_COLS.index("hand_m")])
        return np.column_stack([1 - s, s])


def make_models() -> dict:
    return {
        "hand_threshold": HandThreshold(),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_samples=0.3,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                 class_weight="balanced", random_state=SEED),
    }


def best_f1_threshold(y, score) -> float:
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[np.argmax(f1[:-1])])


def threshold_metrics(y, score, thr) -> dict:
    pred = score >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "csi": tp / max(tp + fp + fn, 1)}  # critical success index (IoU of the flooded class)


def fold_metrics(y, score, thr) -> dict:
    return {"pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
            **threshold_metrics(y, score, thr)}


def fit(model, X, y):
    idx = np.random.default_rng(SEED).permutation(len(y))[:MAX_TRAIN_ROWS]
    return clone(model).fit(X[idx], y[idx])


def tune_threshold(model, X, y, groups) -> float:
    inner_tr, inner_va = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, groups))
    m = fit(model, X[inner_tr], y[inner_tr])
    return best_f1_threshold(y[inner_va], m.predict_proba(X[inner_va])[:, 1])


def cross_validate(model, X, y, groups, splits) -> tuple[dict, np.ndarray]:
    oof = np.full(len(y), np.nan)
    folds = []
    for tr, te in splits:
        thr = tune_threshold(model, X[tr], y[tr], groups[tr])
        oof[te] = fit(model, X[tr], y[tr]).predict_proba(X[te])[:, 1]
        folds.append(fold_metrics(y[te], oof[te], thr))
    df = pd.DataFrame(folds)
    summary = {f"{k}_mean": float(df[k].mean()) for k in df} | {f"{k}_std": float(df[k].std()) for k in df}
    return summary | {"folds": df.round(4).to_dict("records")}, oof


def cross_fitted_calibration(y, oof, fold_of) -> np.ndarray:
    """Isotonic calibration where each fold's scores are mapped by a calibrator fit on the other folds."""
    out = np.empty_like(oof)
    for k in np.unique(fold_of):
        cal = IsotonicRegression(out_of_bounds="clip").fit(oof[fold_of != k], y[fold_of != k])
        out[fold_of == k] = cal.predict(oof[fold_of == k])
    return out


def reliability_table(y, p, bins=10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [{"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int((idx == b).sum()),
             "mean_predicted": float(p[idx == b].mean()), "observed_rate": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def run_cv(df: pd.DataFrame) -> dict:
    X, y = df[FEATURE_COLS].to_numpy(), df[LABEL_COL].to_numpy().astype(int)
    blocks = spatial_blocks(df)
    spatial = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y, blocks))
    random_ = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X))
    fold_of = np.empty(len(y), dtype=int)
    for k, (_, te) in enumerate(spatial):
        fold_of[te] = k

    results, oof_by_model = {}, {}
    for name, model in make_models().items():
        t0 = time.perf_counter()
        sp, oof = cross_validate(model, X, y, blocks, spatial)
        rnd, _ = cross_validate(model, X, y, np.arange(len(y)), random_)
        results[name] = {"spatial_cv": sp, "random_cv": rnd, "seconds": round(time.perf_counter() - t0, 1)}
        oof_by_model[name] = oof
        print(f"{name:24s} spatial PR-AUC {sp['pr_auc_mean']:.3f}±{sp['pr_auc_std']:.3f}  "
              f"random PR-AUC {rnd['pr_auc_mean']:.3f}  ({results[name]['seconds']}s)")
    return {"results": results, "oof": oof_by_model, "fold_of": fold_of, "X": X, "y": y, "blocks": blocks}


def proxy_label_check(df: pd.DataFrame, X, y, blocks) -> dict:
    """How good was the old label? Agreement of the NDWI-change proxy with the GSI extent, and the
    skill of a model trained on the proxy but scored against GSI (same spatial folds)."""
    agreement = threshold_metrics(y, df.proxy_label.to_numpy(), 0.5)
    yp = df.proxy_label.to_numpy().astype(int)
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y, blocks):
        oof[te] = fit(make_models()["random_forest"], X[tr], yp[tr]).predict_proba(X[te])[:, 1]
    return {"proxy_vs_reference": agreement, "proxy_prevalence": float(yp.mean()),
            "rf_trained_on_proxy_scored_on_reference": {"pr_auc": average_precision_score(y, oof),
                                                        "roc_auc": roc_auc_score(y, oof)}}


def train_and_save(df: pd.DataFrame | None = None) -> dict:
    # HistGradientBoosting's OpenMP threads were 28x slower than one thread on an
    # 8-core (4P+4E) laptop (112.6 s vs 4.0 s per fit), so OpenMP is pinned to 1.
    with threadpool_limits(1, user_api="openmp"):
        return _train_and_save(load_table() if df is None else df)


def _train_and_save(df: pd.DataFrame) -> dict:
    cv = run_cv(df)
    X, y, results = cv["X"], cv["y"], cv["results"]
    best = max(results, key=lambda k: results[k]["spatial_cv"]["pr_auc_mean"])

    raw = cv["oof"][best]
    calibrated = cross_fitted_calibration(y, raw, cv["fold_of"])
    threshold = best_f1_threshold(y, calibrated)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    final = fit(make_models()[best], X, y)
    joblib.dump({"model": final, "calibrator": calibrator, "name": best, "features": FEATURE_COLS,
                 "threshold": threshold}, DATA_PROC / "model.joblib")
    risk = calibrator.predict(final.predict_proba(X)[:, 1])
    write_risk_raster(df, {"risk": risk, "risk_oof": calibrated}, threshold)

    report = {
        "selected_model": best,
        "selection_rule": "highest mean spatial-CV PR-AUC",
        "operating_threshold": threshold,
        "cv": results,
        "calibration": {"brier_raw": brier_score_loss(y, raw), "brier_calibrated": brier_score_loss(y, calibrated),
                        "brier_prevalence_only": brier_score_loss(y, np.full(len(y), y.mean())),
                        "reliability": reliability_table(y, calibrated)},
        "proxy_label": proxy_label_check(df, X, y, cv["blocks"]),
        "data": {"n_cells": int(len(y)), "n_flooded": int(y.sum()), "prevalence": float(y.mean()),
                 "n_features": len(FEATURE_COLS), "features": FEATURE_COLS, "block_m": BLOCK_M,
                 "n_blocks": int(len(np.unique(cv["blocks"]))), "n_folds": N_FOLDS,
                 "max_train_rows_per_fit": MAX_TRAIN_ROWS},
        "feature_importance": feature_importance(final, X, y),
        "caveat": CAVEAT,
    }
    (DATA_PROC / "metrics.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    return report


def write_risk_raster(df: pd.DataFrame, scores: dict[str, np.ndarray], threshold: float):
    """Scores back on the analysis grid, so serving is a raster lookup with no model in the request path.

    ``risk`` is the final model (what the API serves); ``risk_oof`` is the cross-fitted out-of-fold
    score, i.e. what the model predicted for each cell without having seen that cell's block.
    """
    grid, _, _ = read_stack(DATA_PROC / "features.tif")
    rows, cols = df.row.to_numpy(), df.col.to_numpy()
    layers = {}
    for name, s in scores.items():
        band = np.full(grid.shape, np.nan, dtype="float32")
        band[rows, cols] = s
        layers[name] = band
    write_stack(DATA_PROC / "risk.tif", grid, layers, tags={"operating_threshold": threshold})


def feature_importance(model, X, y) -> dict:
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), size=min(20000, len(y)), replace=False)
    r = permutation_importance(model, X[idx], y[idx], scoring="average_precision", n_repeats=3,
                               random_state=SEED, n_jobs=-1)
    return dict(sorted(zip(FEATURE_COLS, r.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2, default=float)[:3000])
