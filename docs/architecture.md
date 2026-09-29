# Architecture

```
                 ┌─────────────────────────────────────────────────────┐
                 │                     data/raw/                       │
                 │  terrain (SRTM) · weather (Open-Meteo) · satellite  │
                 │  (Sentinel-2) · gis (river, infra) · client (CSV)   │
                 └───────────────────────┬─────────────────────────────┘
                                          │ src/ingestion/*
                                          │ (schema, CRS, geometry, dup checks)
                                          ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                    src/geo/                         │
                 │  terrain.py   → DEM grid, slope                     │
                 │  indices.py   → river distance, HAND, NDWI/NDVI     │
                 │  weather.py   → IDW rainfall interpolation          │
                 │  zonal.py     → point-in-raster sampling            │
                 │  fusion.py    → joins everything into one grid      │
                 └───────────────────────┬─────────────────────────────┘
                                          │ data/processed/features.csv
                                          ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                    src/ml/                          │
                 │  features.py  → predictor columns, spatial CV groups│
                 │  train.py     → LogisticRegression vs RandomForest, │
                 │                 spatial GroupKFold, metrics.json    │
                 │  model.py     → RiskModel: predict_point/_grid      │
                 │  evaluate.py  → confusion matrix, error_analysis.csv│
                 └───────────────────────┬─────────────────────────────┘
                                          │ data/processed/model.pkl
                                          ▼
        ┌────────────────────┐  ┌────────────────────────────────────┐
        │   src/agents/       │  │            src/api/                │
        │  tools.py  (4 tools)│◄─┤  FastAPI: /predict /assets/exposed  │
        │  agent.py  (LLM loop│  │  /agent/analyze /ingest /grid /...  │
        │   + deterministic   │  │  Pydantic validation, timing        │
        │   fallback)         │  │  middleware, OpenAPI docs           │
        └──────────┬───────────┘  └───────────────┬────────────────────┘
                   │                               │
                   └───────────────┬───────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │   dashboard/app.py       │
                       │  Streamlit, 5 tabs:      │
                       │  map · satellite change  │
                       │  · model perf · client   │
                       │  data/ingest · AI analyst│
                       └─────────────────────────┘
```

## Component responsibilities

- **`src/ingestion`** holds format-agnostic loaders (CSV, GeoJSON/Shapefile, GeoTIFF) behind a small registry (`client_adapter.py`), each returning an `IngestReport` (errors/warnings/metadata) instead of silently failing. Onboarding a new client dataset means calling `ingest_client_dataset(path)`; no new code is required unless the format itself is new.
- **`src/geo`** holds all geospatial transforms: CRS reprojection, raster resampling (`rasterio.warp.reproject`), distance-to-geometry, terrain derivatives (slope, HAND), and the fusion step that joins five data modalities onto one regular grid.
- **`src/ml`** holds feature/label definitions and training with **spatial** cross-validation (`GroupKFold` over coarse lat/lon blocks, not random splits, since adjacent grid cells are spatially autocorrelated and a random split would leak), plus a thin `RiskModel` inference wrapper used by both the API and the agent tools.
- **`src/agents`**: `tools.py` defines four read-only, deterministic functions (region info, point risk, exposed assets, model metrics) as both plain Python callables and Anthropic tool-use JSON schemas. `agent.py` runs a real Claude tool-calling loop when `ANTHROPIC_API_KEY` is set, or a deterministic template narrator when it isn't. Both paths call the same tools and produce numbers from the same deterministic pipeline; only the phrasing differs.
- **`src/api`** wraps the same tools/model used by the agent and dashboard, so all three surfaces (API, agent, dashboard) stay consistent because they share one code path.
- **`dashboard`** imports `src` directly (no network hop) for local iteration speed; in `docker-compose.yml` it instead calls the API service, appropriate for a multi-user deployment.

## Data flow

`fetch_data.py` / `fetch_satellite.py` (one-time, hits public APIs) → `data/raw/*` → `fusion.build_features()` → `data/processed/features.csv` → `train.train_and_save()` → `data/processed/model.pkl` + `metrics.json` → served by API/agent/dashboard. `scripts/run_pipeline.py` runs the middle three steps in one command and completes in under 5 seconds on this AOI size.

## Design decisions and trade-offs

- **Grid-based fusion instead of per-building rasters.** A regular ~650m analysis grid (480 cells) keeps the whole pipeline fast and easy to reason about, at the cost of losing sub-grid detail, acceptable for regional triage and documented as a limitation for parcel-level decisions.
- **Proxy ML label instead of a fabricated one.** No free, reproducible historical flood-extent polygon dataset was available for this AOI. Rather than inventing performance numbers against a synthetic label, the label is a real observational signal (Sentinel-2 NDWI increase, pre vs. post flood), genuine remote sensing but a weak proxy for actual inundation given the cloud-cover limitation above. This is stated everywhere the model's output appears.
- **Deterministic tools, LLM narration only.** The agent never computes a number; every statistic in a report traces back to a tool call, logged in `tool_log`. This is both a safety property (no hallucinated numbers) and a debuggability property (you can replay any report from its tool log).
- **No message-passing queue or database.** For a single-AOI PoC, flat files (`features.csv`, `model.pkl`) are simpler and fully reproducible from `run_pipeline.py`; a real deployment covering many AOIs/clients would need a proper store (see Future Work in the README).

## Scalability considerations / failure points

- `RiskModel` loads `features.csv` fully into memory per API worker; fine at 480 rows, would need chunking/tiling for a country-scale grid.
- The deterministic NLU in `agent.py` (`parse_query`) is keyword/regex based and only knows the fixed `PLACES` dict for this AOI; it does not generalize to new regions without either adding places or wiring the real LLM path (which does generalize, since Claude parses free text itself).
- `opentopodata` and `open-meteo` are rate-limited public APIs; `fetch_data.py` caches results to disk and is idempotent, but a multi-AOI production ingestion job would need retry/backoff and a paid tier or self-hosted SRTM tiles.
- The API holds one `RiskModel` singleton in process memory (`src/api/main.py:rm()`); horizontal scaling would need either a shared model-serving layer or per-worker reload logic.
