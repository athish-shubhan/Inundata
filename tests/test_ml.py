import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.config import DATA_PROC
from src.ml.features import BLOCK_M, FEATURE_COLS, LABEL_COL, load_table, spatial_blocks
from src.ml.model import RiskModel
from src.ml.train import HandThreshold, best_f1_threshold, cross_fitted_calibration, threshold_metrics

DOWNTOWN = (32.2158, 130.7535)


@pytest.fixture(scope="module")
def table():
    return load_table()


@pytest.fixture(scope="module")
def rm():
    return RiskModel()


@pytest.fixture(scope="module")
def metrics():
    return json.loads((DATA_PROC / "metrics.json").read_text())


def test_no_post_event_information_in_predictors():
    leaky = {"flooded", "flood_fraction", "ndwi_change", "proxy_label", "cloud"}
    assert not leaky & set(FEATURE_COLS)
    assert not any("post" in c for c in FEATURE_COLS)
    assert LABEL_COL not in FEATURE_COLS


def test_table_drops_water_and_cloud(table):
    assert len(table) > 100_000
    assert (table.permanent_water == 0).all() and (table.cloud == 0).all()
    assert table[FEATURE_COLS].notna().all().all()
    assert set(table[LABEL_COL].unique()) == {0, 1}


def test_spatial_blocks_are_square_metric_tiles(table):
    b = spatial_blocks(table)
    assert len(b) == len(table) and len(np.unique(b)) > 10
    same = table[b == b[0]]
    assert same.x.max() - same.x.min() < BLOCK_M and same.y.max() - same.y.min() < BLOCK_M


def test_best_f1_threshold_separates_clean_data():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_f1_threshold(y, s)
    assert 0.3 < t <= 0.8
    assert threshold_metrics(y, s, t) == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "csi": 1.0}


def test_cross_fitted_calibration_stays_in_unit_interval():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    raw = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 500), 0, 1)
    cal = cross_fitted_calibration(y, raw, np.arange(500) % 5)
    assert cal.min() >= 0 and cal.max() <= 1
    assert cal[y == 1].mean() > cal[y == 0].mean()


def test_hand_threshold_baseline_is_monotone():
    X = np.zeros((3, len(FEATURE_COLS)))
    X[:, FEATURE_COLS.index("hand_m")] = [0, 5, 50]
    p = HandThreshold().fit(X, None).predict_proba(X)[:, 1]
    assert p[0] > p[1] > p[2]


def test_metrics_report_selection_and_both_cv_schemes(metrics):
    best = metrics["selected_model"]
    sp = {k: v["spatial_cv"]["pr_auc_mean"] for k, v in metrics["cv"].items()}
    assert best == max(sp, key=sp.get)
    assert "random_cv" in metrics["cv"][best]
    assert metrics["cv"][best]["spatial_cv"]["pr_auc_mean"] > metrics["cv"]["hand_threshold"]["spatial_cv"]["pr_auc_mean"]
    assert metrics["caveat"]


def test_predict_point_downtown(rm):
    r = rm.predict_point(*DOWNTOWN)
    assert r["status"] == "scored" and 0 <= r["risk_score"] <= 1
    assert set(FEATURE_COLS) == set(r["drivers"])
    assert r["risk_band"] in {"LOW", "MODERATE", "HIGH"}


def test_predict_point_outside_aoi_raises(rm):
    with pytest.raises(ValueError):
        rm.predict_point(35.0, 139.0)


def test_predict_points_matches_point_lookup(rm):
    pts = pd.DataFrame({"lat": [DOWNTOWN[0], 32.2369, 35.0], "lon": [DOWNTOWN[1], 130.6948, 139.0]})
    out = rm.predict_points(pts)
    assert out.risk_score.iloc[0] == pytest.approx(rm.predict_point(*DOWNTOWN)["risk_score"])
    assert np.isnan(out.risk_score.iloc[2]) and out.risk_band.iloc[2] == "UNSCORED"


def test_risk_raster_is_calibrated_probability(rm, table):
    s = rm.risk[table.row, table.col]
    assert np.isfinite(s).all() and s.min() >= 0 and s.max() <= 1
    assert abs(s.mean() - table[LABEL_COL].mean()) < 0.01  # calibrated: mean score matches the flooded share


def test_served_scores_do_not_replay_training_labels(rm, table, metrics):
    """The served map must be out-of-fold. Scoring with the final model gave pooled PR-AUC 0.92,
    well above the 0.82 spatial-CV estimate, because it had seen these cells' labels."""
    served = average_precision_score(table[LABEL_COL], rm.risk[table.row, table.col])
    assert served <= metrics["cv"][metrics["selected_model"]]["spatial_cv"]["pr_auc_mean"] + 0.03


def test_driver_percentiles(rm):
    low = rm.predict_point(32.2369, 130.6948)["driver_percentiles"]  # valley floor at Watari
    assert all(0 <= v <= 100 for v in low.values())
    assert rm.percentile("elev_m", -1e9) == 0 and rm.percentile("elev_m", 1e9) == 100


def test_coarse_grid(rm):
    g = rm.coarse_grid(10)
    assert 500 < len(g) < 2000
    assert g.risk_score.between(0, 1).all()
