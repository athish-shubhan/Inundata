import joblib
import pandas as pd
from src.config import DATA_PROC
from src.ml.features import FEATURE_COLS

class RiskModel:
    def __init__(self):
        d = joblib.load(DATA_PROC / "model.pkl")
        self.model, self.name = d["model"], d["name"]
        self.features = pd.read_csv(DATA_PROC / "features.csv")

    def predict_grid(self) -> pd.DataFrame:
        df = self.features.copy()
        df["risk_score"] = self.model.predict_proba(df[FEATURE_COLS].values)[:, 1]
        return df

    def nearest_row(self, lat, lon) -> pd.Series:
        d2 = (self.features.lat - lat) ** 2 + (self.features.lon - lon) ** 2
        return self.features.loc[d2.idxmin()]

    def predict_point(self, lat, lon) -> dict:
        row = self.nearest_row(lat, lon)
        x = row[FEATURE_COLS].values.reshape(1, -1)
        proba = float(self.model.predict_proba(x)[0, 1])
        return {"lat": lat, "lon": lon, "risk_score": proba,
                "nearest_grid_cell": {"lat": float(row.lat), "lon": float(row.lon)},
                "drivers": {c: float(row[c]) for c in FEATURE_COLS}}

    def predict_points(self, df_pts: pd.DataFrame, lat_col="lat", lon_col="lon") -> pd.DataFrame:
        out = df_pts.copy()
        scores = []
        for _, r in out.iterrows():
            row = self.nearest_row(r[lat_col], r[lon_col])
            x = row[FEATURE_COLS].values.reshape(1, -1)
            scores.append(float(self.model.predict_proba(x)[0, 1]))
        out["risk_score"] = scores
        return out
