import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.agents import tools
from src.agents.agent import run_agent
from src.api.schemas import (AgentRequest, AgentResponse, ExposedAssetsResponse, ExposedInfrastructureResponse,
                             IngestResponse, PredictResponse)
from src.config import CORS_ORIGINS, DATA_PROC
from src.ingestion.client_adapter import ingest_client_dataset
from src.services.logging import get_logger

log = get_logger("api")
ARTIFACTS = [DATA_PROC / f for f in ("features.tif", "risk.tif", "metrics.json")]


def model_ready() -> bool:
    return all(p.exists() for p in ARTIFACTS)


def require_model():
    if not model_ready():
        raise HTTPException(503, "model not trained yet, run scripts/run_pipeline.py")
    return tools._model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the rasters at startup so the first request does not pay for it.
    if model_ready():
        tools._model()
    yield


app = FastAPI(title="Inundata API", version="0.2.0", lifespan=lifespan,
              description="Flood-susceptibility API for the Kuma River basin: 30 m terrain/land-cover model "
                          "validated against the observed July 2020 inundation, plus an agentic analyst.")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def timing(request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    dt = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"{request.method} {request.url.path} {resp.status_code} {dt}ms")
    resp.headers["X-Process-Time-Ms"] = str(dt)
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model_ready()}


@app.get("/region")
def region():
    return tools.get_region_info()


@app.get("/predict", response_model=PredictResponse)
def predict(lat: float, lon: float):
    try:
        return require_model().predict_point(lat, lon)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/assets/exposed", response_model=ExposedAssetsResponse)
def exposed_assets(lat: float, lon: float, radius_km: float = Query(2.0, gt=0, le=20)):
    require_model()
    return tools.list_exposed_assets(lat, lon, radius_km)


@app.get("/infrastructure/exposed", response_model=ExposedInfrastructureResponse)
def exposed_infrastructure(lat: float, lon: float, radius_km: float = Query(2.0, gt=0, le=20)):
    require_model()
    return tools.list_exposed_infrastructure(lat, lon, radius_km)


@app.get("/rainfall")
def rainfall():
    return tools.get_event_rainfall()


@app.get("/model/metrics")
def metrics():
    if not (DATA_PROC / "metrics.json").exists():
        raise HTTPException(503, "metrics not available, train the model first")
    return tools.get_model_metrics()


@app.get("/grid")
def grid(step: int = Query(10, ge=1, le=50, description="block size in 30 m cells; step=10 returns 300 m blocks")):
    df = require_model().coarse_grid(step)
    return df.round(4).to_dict("records")


@app.post("/agent/analyze", response_model=AgentResponse)
def analyze(req: AgentRequest):
    require_model()
    return run_agent(req.query)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), lat_col: str = "lat", lon_col: str = "lon"):
    # The loaders pick a parser from the extension, so keep the uploaded file's suffix.
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "").suffix, delete=False) as tmp:
        tmp.write(await file.read())
    try:
        result = ingest_client_dataset(tmp.name, lat_col=lat_col, lon_col=lon_col)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    rep = result.report
    rep.source = file.filename or "upload"
    return rep.to_dict()
