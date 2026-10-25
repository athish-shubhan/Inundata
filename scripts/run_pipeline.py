"""features.tif -> spatial CV + model selection -> risk.tif / metrics.json -> out-of-fold error analysis.

Raw inputs must already be in data/<aoi>/raw/ (python scripts/fetch_data.py).

    python scripts/run_pipeline.py              # everything
    python scripts/run_pipeline.py --skip-features
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.geo.fusion import build_features
from src.ml.evaluate import error_report
from src.ml.train import train_and_save
from src.services.logging import get_logger

log = get_logger("pipeline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-features", action="store_true", help="reuse the existing features.tif")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if not args.skip_features:
        info = build_features()
        log.info(f"features: {info['shape']} grid, {info['flooded_cells']} flooded cells, {info['seconds']}s")

    t1 = time.perf_counter()
    report = train_and_save()
    best = report["selected_model"]
    sp = report["cv"][best]["spatial_cv"]
    log.info(f"selected {best} in {time.perf_counter() - t1:.0f}s: spatial-CV PR-AUC {sp['pr_auc_mean']:.3f}, "
             f"CSI {sp['csi_mean']:.3f}")

    err = error_report()
    log.info(f"out-of-fold confusion matrix at threshold {err['threshold']:.2f}: {err['confusion_matrix']}")
    log.info(f"pipeline complete in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
