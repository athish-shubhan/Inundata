import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src.agents import tools
from src.agents.agent import run_agent
from src.api.schemas import (AgentRequest, AgentResponse, ExposedAssetsResponse, ExposedInfrastructureResponse,
                             IngestResponse, PredictResponse)
from src.config import CORS_ORIGINS, DATA_PROC
from src.ingestion.client_adapter import LOADERS, ingest_client_dataset
from src.services.logging import get_logger

log = get_logger("api")
Lat = Annotated[float, Query(ge=-90, le=90, allow_inf_nan=False, description="WGS84 latitude")]
Lon = Annotated[float, Query(ge=-180, le=180, allow_inf_nan=False, description="WGS84 longitude")]
Radius = Annotated[float, Query(gt=0, le=tools.MAX_RADIUS_KM, description="search radius in km")]
ARTIFACTS = [DATA_PROC / f for f in ("features.tif", "risk.tif", "metrics.json")]
MAX_UPLOAD_BYTES = 50 * 2**20  # client registers are kilobytes to a few MB; this bounds disk use per request
UPLOAD_CHUNK_BYTES = 2**20


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
def predict(lat: Lat, lon: Lon):
    try:
        return require_model().predict_point(lat, lon)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/assets/exposed", response_model=ExposedAssetsResponse)
def exposed_assets(lat: Lat, lon: Lon, radius_km: Radius = 2.0):
    require_model()
    return tools.list_exposed_assets(lat, lon, radius_km)


@app.get("/infrastructure/exposed", response_model=ExposedInfrastructureResponse)
def exposed_infrastructure(lat: Lat, lon: Lon, radius_km: Radius = 2.0):
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
    # The loaders pick a parser from the extension, so check it before reading the body and keep it on disk.
    suffix = Path(file.filename or "").suffix
    if suffix.lstrip(".").lower() not in LOADERS:
        raise HTTPException(400, f"unsupported format '{suffix or '(none)'}'. supported: {sorted(LOADERS)}")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = Path(tmp.name)
    try:
        size = 0
        with path.open("wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"upload larger than {MAX_UPLOAD_BYTES // 2**20} MB")
                out.write(chunk)
        # geopandas/rasterio parsing is blocking; keep it off the event loop.
        result = await run_in_threadpool(ingest_client_dataset, path, lat_col=lat_col, lon_col=lon_col)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        path.unlink(missing_ok=True)
    rep = result.report
    rep.source = file.filename or "upload"
    # Loader messages quote the file they read, which here is the server's temp file; show the upload's name.
    rep.errors = [m.replace(str(path), rep.source) for m in rep.errors]
    rep.warnings = [m.replace(str(path), rep.source) for m in rep.warnings]
    return rep.to_dict()
