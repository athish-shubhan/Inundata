from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

AOI_NAME = "Hitoyoshi, Kumamoto, Japan"
LAT_MIN, LAT_MAX = 32.74, 32.86
LON_MIN, LON_MAX = 130.68, 130.82
GRID_STEP = 0.006
CRS = "EPSG:4326"
METRIC_CRS = "EPSG:6690"

FLOOD_EVENT_START = "2020-07-01"
FLOOD_EVENT_END = "2020-07-10"
BASELINE_START = "2020-06-01"
BASELINE_END = "2020-06-10"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")
