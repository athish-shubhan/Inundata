import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# The study area is data, not code: point INUNDATA_AOI at another JSON file
# with the same fields to run the pipeline somewhere else.
AOI_PATH = Path(os.getenv("INUNDATA_AOI", ROOT / "config" / "hitoyoshi.json"))
AOI = json.loads(AOI_PATH.read_text())

AOI_ID = AOI["id"]
AOI_NAME = AOI["name"]
LAT_MIN, LAT_MAX = AOI["bbox"]["lat_min"], AOI["bbox"]["lat_max"]
LON_MIN, LON_MAX = AOI["bbox"]["lon_min"], AOI["bbox"]["lon_max"]
CRS = "EPSG:4326"
METRIC_CRS = AOI["metric_crs"]
RESOLUTION_M = AOI["resolution_m"]

DATA_RAW = ROOT / "data" / AOI_ID / "raw"
DATA_PROC = ROOT / "data" / AOI_ID / "processed"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")
CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if o]
