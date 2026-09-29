import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.services.logging import get_logger
from src.geo.fusion import build_features
from src.ml.train import train_and_save
from src.ml.evaluate import error_report

log = get_logger("pipeline")

def main():
    t0 = time.time()
    log.info("building multimodal feature table...")
    df = build_features()
    log.info(f"features: {df.shape} rows in {time.time()-t0:.1f}s")

    t1 = time.time()
    report = train_and_save()
    log.info(f"trained {report['selected_model']} in {time.time()-t1:.1f}s, "
              f"cv_f1={report['cv_metrics'][report['selected_model']]['f1']:.3f}")

    err = error_report()
    log.info(f"error analysis: confusion_matrix={err['confusion_matrix']}")
    log.info(f"pipeline complete in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
