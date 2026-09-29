import time
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.config import DATA_PROC
from src.agents.tools import get_region_info, assess_region, list_exposed_assets, get_model_metrics
from src.agents.agent import run_agent
from src.ingestion.client_adapter import ingest_client_dataset
from src.ml.model import RiskModel
from src.services.logging import get_logger
from src.api.schemas import PredictResponse, ExposedAssetsResponse, AgentRequest, AgentResponse, IngestResponse

log = get_logger("api")
app = FastAPI(title="Inundata API", version="0.1.0",
              description="Flood-risk PoC for the Kuma River basin: geospatial + ML + agentic analysis.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_rm: RiskModel | None = None

def rm() -> RiskModel:
    global _rm
    if _rm is None:
        if not (DATA_PROC / "model.pkl").exists():
            raise HTTPException(503, "model not trained yet, run scripts/run_pipeline.py")
        _rm = RiskModel()
    return _rm

@app.middleware("http")
async def timing(request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    dt = round((time.time() - t0) * 1000, 1)
    log.info(f"{request.method} {request.url.path} {resp.status_code} {dt}ms")
    resp.headers["X-Process-Time-Ms"] = str(dt)
    return resp

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (DATA_PROC / "model.pkl").exists()}

@app.get("/region")
def region():
    return get_region_info()

@app.get("/predict", response_model=PredictResponse)
def predict(lat: float, lon: float):
    rm()
    return assess_region(lat, lon)

@app.get("/assets/exposed", response_model=ExposedAssetsResponse)
def exposed(lat: float, lon: float, radius_km: float = 2.0):
    rm()
    return list_exposed_assets(lat, lon, radius_km)

@app.get("/model/metrics")
def metrics():
    if not (DATA_PROC / "metrics.json").exists():
        raise HTTPException(503, "metrics not available, train the model first")
    return get_model_metrics()

@app.get("/grid")
def grid():
    df = rm().predict_grid()
    cols = ["lat", "lon", "risk_score", "elev_m", "hand_m", "river_dist_m", "rain_flood_mm"]
    return df[cols].round(4).to_dict("records")

@app.post("/agent/analyze", response_model=AgentResponse)
def analyze(req: AgentRequest):
    rm()
    return run_agent(req.query)

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), lat_col: str = "lat", lon_col: str = "lon"):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = ingest_client_dataset(tmp_path, lat_col=lat_col, lon_col=lon_col)
    except ValueError as e:
        raise HTTPException(400, str(e))
    rep = result.report if hasattr(result, "report") else result[1]
    rep.source = file.filename
    Path(tmp_path).unlink(missing_ok=True)
    return rep.to_dict()
