import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.config import DATA_PROC
from src.ml.features import FEATURE_COLS, spatial_groups, Xy

MODELS = {
    "logistic_regression": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))]),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=0),
}

def spatial_cv(df: pd.DataFrame, name: str, n_splits=4):
    X, y = Xy(df)
    groups = spatial_groups(df)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    prec, rec, f1, auc = [], [], [], []
    for tr, te in gkf.split(X, y, groups):
        if y[te].sum() == 0 or y[tr].sum() == 0:
            continue
        m = MODELS[name]
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        prec.append(precision_score(y[te], pred, zero_division=0))
        rec.append(recall_score(y[te], pred, zero_division=0))
        f1.append(f1_score(y[te], pred, zero_division=0))
        try:
            auc.append(roc_auc_score(y[te], proba))
        except ValueError:
            pass
    return {"precision": float(np.mean(prec)), "recall": float(np.mean(rec)),
            "f1": float(np.mean(f1)), "roc_auc": float(np.mean(auc)) if auc else None, "n_folds_used": len(prec)}

def train_and_save():
    df = pd.read_csv(DATA_PROC / "features.csv")
    metrics = {name: spatial_cv(df, name) for name in MODELS}
    best = max(metrics, key=lambda k: metrics[k]["f1"])
    X, y = Xy(df)
    final = MODELS[best]
    final.fit(X, y)
    joblib.dump({"model": final, "name": best, "features": FEATURE_COLS}, DATA_PROC / "model.pkl")

    if best == "random_forest":
        importance = dict(zip(FEATURE_COLS, final.feature_importances_.tolist()))
    else:
        importance = dict(zip(FEATURE_COLS, final.named_steps["clf"].coef_[0].tolist()))
    report = {"cv_metrics": metrics, "selected_model": best, "n_samples": len(df),
              "n_positive": int(y.sum()), "feature_importance": importance,
              "caveat": "labels are a proxy derived from Sentinel-2 NDWI change detection (post minus pre-flood), not verified historical flood-extent ground truth; small, spatially imbalanced sample -> treat scores as directional, not calibrated probabilities."}
    (DATA_PROC / "metrics.json").write_text(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    r = train_and_save()
    print(json.dumps(r, indent=2))
