# Architecture

```
                 ┌─────────────────────────────────────────────────────┐
                 │   config/hitoyoshi.json  (bbox, CRS, 30 m, dates,   │
                 │   scene URLs, GSI tiles, named places)              │
                 └───────────────────────┬─────────────────────────────┘
                                          │ scripts/fetch_data.py (idempotent, retry/backoff)
                                          ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                data/hitoyoshi/raw/                  │
                 │  terrain (Copernicus DEM) · satellite (Sentinel-2)  │
                 │  reference (GSI flood mask) · gis (OSM river,       │
                 │  infrastructure) · weather · client portfolio       │
                 └───────────────────────┬─────────────────────────────┘
                                          ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                    src/geo/                         │
                 │  grid.py      → one 30 m UTM grid, warp, stack I/O  │
                 │  terrain.py   → DEM, slope, TPI, river mask/dist    │
                 │  hydrology.py → fill, D8, flow accumulation, HAND   │
                 │  indices.py   → NDWI/NDVI/NDBI, SCL cloud mask      │
                 │  weather.py   → event rainfall (context only)       │
                 │  fusion.py    → predictors + label + diagnostics    │
                 └───────────────────────┬─────────────────────────────┘
                                          │ processed/features.tif (15 named bands)
                                          ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                    src/ml/                          │
                 │  features.py  → predictor list, scorable cells,     │
                 │                 2 km spatial blocks                 │
                 │  train.py     → 4 models, spatial + random CV,      │
                 │                 calibration, proxy-label check      │
                 │  evaluate.py  → out-of-fold error analysis          │
                 │  model.py     → RiskModel: raster lookups           │
                 └───────────────────────┬─────────────────────────────┘
                                          │ processed/risk.tif · metrics.json · model.joblib
                                          ▼
        ┌────────────────────┐  ┌────────────────────────────────────┐
        │   src/agents/       │  │            src/api/                │
        │  tools.py  (6 tools)│◄─┤  FastAPI: /predict /assets/exposed  │
        │  agent.py  (LLM loop│  │  /infrastructure/exposed /grid ...  │
        │   + deterministic   │  │  Pydantic validation, timing        │
        │   fallback)         │  │  middleware, OpenAPI docs           │
        └──────────┬───────────┘  └───────────────┬────────────────────┘
                   └───────────────┬───────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │   dashboard/app.py       │
                       │  Streamlit, 5 tabs       │
                       └─────────────────────────┘
```

## Component responsibilities

- **`config/`** holds the study area as data. `src/config.py` reads the file named by `INUNDATA_AOI` (default `config/hitoyoshi.json`) and derives every path from its `id`, so a second AOI is a second JSON file, not a code change.
- **`src/ingestion`** holds format-agnostic loaders (CSV, GeoJSON/Shapefile, GeoTIFF) behind a small registry (`client_adapter.py`), each returning an `IngestReport` (errors/warnings/metadata) instead of failing on the first problem.
- **`src/geo`** does all spatial work on one grid. `grid.analysis_grid()` snaps the AOI to a 30 m UTM raster; `warp()` puts any input on it with an explicit resampling choice; `write_stack`/`read_stack` keep every layer in one named-band GeoTIFF. `hydrology.py` routes flow on a DEM padded by 100 cells so catchments are not cut at the AOI edge, then crops back.
- **`src/ml`** turns the stack into a table of scorable cells, cross-validates over 2 km blocks, calibrates out-of-fold scores, and writes the selected model's scores back to the grid as `risk.tif` (`risk` from the final model, `risk_oof` cross-fitted).
- **`src/agents`**: `tools.py` defines six read-only functions as plain Python callables and Anthropic tool-use JSON schemas. `agent.py` runs a Claude tool-calling loop when `ANTHROPIC_API_KEY` is set, or a deterministic template narrator when it isn't; both call the same tools.
- **`src/api`** wraps the same tools, so the API, agent and dashboard share one code path and one `RiskModel` instance per process.
- **`dashboard`** imports `src` directly (no network hop) for local iteration speed; `docker-compose.yml` runs the API and dashboard as separate services.

## Data flow

`fetch_data.py` (one-time, public endpoints) → `data/hitoyoshi/raw/*` → `fusion.build_features()` → `features.tif` → `train.train_and_save()` → `model.joblib`, `risk.tif`, `metrics.json` → `evaluate.error_report()` → `error_analysis.csv`. `scripts/run_pipeline.py` runs everything after the fetch in about 2.5 minutes.

## Design decisions and trade-offs

- **Score once, serve a raster.** All 153k scorable cells are scored at training time, so a request is a coordinate transform and an array index. No sklearn model is loaded in the API, cold start is a ~6 MB raster read, and scoring a portfolio is vectorised (`RiskModel.predict_points`).
- **A real label, and a measured comparison with the proxy.** The GSI estimate is the closest thing to ground truth that is public for this event. Keeping the old NDWI-change label as a diagnostic band lets the pipeline report how poor it was (CSI 0.13 against GSI) instead of asserting it.
- **Spatial CV everywhere a number is reported**, including threshold tuning (inner split of training blocks) and calibration (cross-fitted), with random CV alongside to make the leakage visible.
- **Pure-numpy hydrology.** Priority-Flood and D8 are a few dozen lines; at ~360k cells they run in seconds, so a compiled dependency (pysheds, richdem, numba) is not worth the install footprint.
- **Deterministic tools, LLM narration only.** The agent never computes a number; every statistic in a report traces back to a logged tool call.
- **Flat files, no database.** For a single AOI, GeoTIFF + JSON artifacts are simpler and fully reproducible; several AOIs or clients would need a tiled store and a model registry.

## Scalability considerations / failure points

- `RiskModel` holds the full feature and risk stacks in memory (~10 MB here). A national grid would need tiled/COG-backed windowed reads instead.
- `hydrology.py` is single-threaded Python loops over a heap; it is fine at this size but would need a compiled implementation for large basins.
- Overpass, Open-Meteo and the GSI tile server are public and rate-limited; `fetch_data.py` retries with backoff and skips files that already exist, but a production job would need a mirrored or paid source.
- The deterministic NLU (`parse_query`) only knows the places in the AOI config; the LLM path parses free text.
- One `RiskModel` per worker process (`src/agents/tools.py:_model`); horizontal scaling just repeats the raster load per worker.
