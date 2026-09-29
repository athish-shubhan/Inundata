import pandas as pd
from src.config import DATA_PROC
from src.ml.features import spatial_groups, FEATURE_COLS, LABEL_COL
from src.ml.model import RiskModel

def _features_df():
    return pd.read_csv(DATA_PROC / "features.csv")

def test_features_file_has_required_columns():
    df = _features_df()
    for c in FEATURE_COLS + [LABEL_COL]:
        assert c in df.columns
    assert len(df) > 100

def test_spatial_groups_multiple_blocks():
    df = _features_df()
    g = spatial_groups(df, n_blocks=4)
    assert len(set(g)) > 1
    assert len(g) == len(df)

def test_model_metrics_present():
    import json
    m = json.loads((DATA_PROC / "metrics.json").read_text())
    assert "cv_metrics" in m and "selected_model" in m
    assert m["selected_model"] in m["cv_metrics"]

def test_risk_model_predict_point_in_range():
    rm = RiskModel()
    r = rm.predict_point(32.795, 130.755)
    assert 0.0 <= r["risk_score"] <= 1.0
    assert set(FEATURE_COLS) <= set(r["drivers"].keys())

def test_risk_model_predict_grid_scores():
    rm = RiskModel()
    df = rm.predict_grid()
    assert "risk_score" in df.columns
    assert df.risk_score.between(0, 1).all()
