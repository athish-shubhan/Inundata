from sklearn.metrics import confusion_matrix, classification_report
from src.config import DATA_PROC
from src.ml.model import RiskModel
from src.ml.features import LABEL_COL

def error_report():
    rm = RiskModel()
    df = rm.predict_grid()
    pred = (df.risk_score >= 0.5).astype(int)
    cm = confusion_matrix(df[LABEL_COL], pred).tolist()
    report = classification_report(df[LABEL_COL], pred, zero_division=0, output_dict=True)
    df["pred"] = pred
    df["error_type"] = "correct"
    df.loc[(df[LABEL_COL] == 1) & (pred == 0), "error_type"] = "false_negative"
    df.loc[(df[LABEL_COL] == 0) & (pred == 1), "error_type"] = "false_positive"
    df[df.error_type != "correct"][["lat", "lon", LABEL_COL, "risk_score", "error_type"]].to_csv(DATA_PROC / "error_analysis.csv", index=False)
    return {"confusion_matrix": cm, "report": report}

if __name__ == "__main__":
    import json
    print(json.dumps(error_report(), indent=2))
