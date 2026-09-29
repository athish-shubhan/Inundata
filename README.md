# Inundata

An independent geospatial flood-intelligence platform. It takes a real disaster (the July 2020 Kuma River flood in Hitoyoshi, Kumamoto, Japan), fuses satellite, weather, terrain, and GIS data, trains a flood-susceptibility model, and exposes it through an agentic natural-language interface, a REST API, and a client-facing dashboard.

> **The client engagement described below is simulated**, built to work through the full problem end to end. The elevation, rainfall, and satellite imagery are real, public data for a real event; the client portfolio and infrastructure register are clearly-labeled synthetic data standing in for what a real client would provide.

---

## 1. Problem

A regional insurer/infrastructure operator in Kumamoto Prefecture wants to know, after a heavy rainfall event, which of its properties and which critical infrastructure sit in flood-prone terrain along the Kuma River — fast, and in language a risk manager (not a GIS analyst) can act on.

## 2. Client (simulated)

**Client:** a mid-size property & casualty insurer with a portfolio of 120 residential/commercial/industrial/agricultural assets in Hitoyoshi City.
**Business problem:** rapidly triage flood exposure after a rainfall event, without a dedicated GIS team.
**Client data:** a CSV asset register (`data/raw/client/client_portfolio.csv`) — lat/lon, building type, insured value, year built, occupants.
**External data:** SRTM elevation, Open-Meteo rainfall, Sentinel-2 imagery, a digitized river centerline.
**Question:** *"Assess the flood risk around this region after recent rainfall and identify exposed infrastructure."*
**Deliverable:** a client-facing markdown risk report, an interactive dashboard, and an API other systems can call.
**Success criteria:** an analyst can go from a plain-English question to a ranked, explainable list of at-risk assets in under a second of model inference time.

## 3. Solution

```
data ingestion → geospatial fusion → ML risk model → agentic analysis → API → dashboard
```

Five data modalities (terrain, weather, satellite, GIS/vector, client) are fused onto one 480-cell analysis grid over the Kuma River basin. A Random Forest trained on that grid predicts a flood-susceptibility score; an LLM-capable agent (with a fully-functional deterministic fallback) turns a plain-English request into a structured, honest client report by calling deterministic tools — it never computes a number itself.

## 4. Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full component diagram, data flow, and trade-offs. Short version: `src/ingestion` → `src/geo` → `src/ml` → `src/agents` / `src/api` → `dashboard/`, all sharing the same model and tool code so the three surfaces never disagree.

## 5. Data

See [`docs/data_sources.md`](docs/data_sources.md) for the full source/license/resolution/limitations table. Summary:

- **Elevation:** SRTM 90m, real, via OpenTopoData.
- **Weather:** real daily rainfall, Open-Meteo historical archive, 3 stations IDW-interpolated (2020-07-01–10 flood window vs. a 2020-06-01–10 baseline).
- **Satellite:** real Sentinel-2 L2A surface reflectance (bands B03/B04/B08/B11 + SCL), pre-flood (2020-05-11) and post-flood (2020-08-29), pulled directly from the public `sentinel-cogs` bucket via windowed COG reads — no download of full tiles.
- **GIS:** a simplified, hand-digitized Kuma River centerline (Overpass/OSM was unreachable from this sandbox — documented limitation) and a simulated critical-infrastructure register.
- **Client:** a simulated 120-asset property portfolio.

## 6. Geospatial processing

- CRS reprojection between WGS84 (`EPSG:4326`), UTM 52N (Sentinel-2 native), and a metric CRS (`EPSG:6690`) for distance calculations.
- Windowed, on-the-fly reprojection of cloud-optimized GeoTIFFs (`rasterio.warp.reproject`) instead of full-tile downloads.
- Geometry validation and auto-repair (`buffer(0)`) in the ingestion layer.
- Distance-to-river and Height-Above-Nearest-Drainage (HAND) — a standard hydrological terrain proxy for flood susceptibility.
- Zonal/neighborhood statistics (3×3 window mean and local relief) — real spatial aggregation, not just raw lat/lon.
- IDW spatial interpolation of point rainfall observations onto the analysis grid.

## 7. Remote sensing

NDWI and NDVI computed from real Sentinel-2 reflectance; NDWI change detection (post-flood minus pre-flood) is used as an inundation-signal proxy. SCL-based cloud masking is implemented (`src/geo/indices.py:cloud_mask`). The dashboard's "Satellite Change Detection" tab documents *why* this is a proxy and not verified flood extent (optical can't see through the storm clouds present at the actual flood peak).

## 8. Multimodal fusion

`src/geo/fusion.py` builds one feature table combining: elevation, slope, HAND, river distance (terrain) · rainfall + rainfall anomaly (weather, IDW-interpolated) · NDVI/NDWI + NDWI change (satellite) · neighborhood elevation/HAND statistics (spatial context) · the resulting `inundation_signal` label. Alignment: all layers are resampled/reprojected onto the same 0.006° analysis grid; missing/out-of-bounds cells are dropped (480 of 480 grid cells retained in this AOI).

## 9. Machine learning

Two models are compared with **spatial** `GroupKFold` cross-validation (not random splits — adjacent grid cells are spatially autocorrelated):

| model | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| logistic_regression | 0.20 | 0.97 | 0.30 | 0.82 |
| **random_forest (selected)** | 0.36 | 0.37 | **0.32** | **0.93** |

*(regenerate with `python scripts/run_pipeline.py`; numbers above are from this repo's own last run, not fabricated.)*

**Honest caveat, stated everywhere this model's output appears:** the label is a proxy (Sentinel-2 NDWI increase > 0.04, pre vs. post flood), not a verified historical flood-extent inventory, and the positive class is small (21/480 cells). Treat scores as directional risk ranking, not calibrated probabilities. Error analysis (`src/ml/evaluate.py`, in-sample) is used only to inspect *where* the model disagrees, not to claim generalization — the CV table above is the honest generalization estimate.

## 10. Spatial AI

Beyond raw coordinates: Height-Above-Nearest-Drainage, geodesic distance-to-river, 3×3 neighborhood mean/relief of elevation and HAND, and spatially-grouped cross-validation blocks so evaluation respects spatial autocorrelation instead of leaking between neighboring cells.

## 11. Agentic workflow

`src/agents/agent.py` implements a real Claude tool-calling loop (`llm_report`) using four read-only tools (`get_region_info`, `assess_region`, `list_exposed_assets`, `get_model_metrics`, defined in `src/agents/tools.py` with JSON schemas) plus a fully-functional **deterministic fallback** (`deterministic_report`) used automatically when `ANTHROPIC_API_KEY` is unset — which is how this repo runs out of the box, and how it was verified end-to-end (see §17). Both paths call the exact same tools and produce identical numbers; only the report's phrasing differs. Every tool call is logged with latency (`tool_log`), the LLM loop is capped at 8 tool calls, and the agent's toolset is a fixed whitelist of read-only functions — it cannot execute shell commands or write files.

Example query the agent handles: *"Assess the flood risk around downtown Hitoyoshi after recent rainfall and identify exposed infrastructure within 2km."*

## 12. Dashboard

Streamlit, five tabs: **Risk Map** (folium, risk heatmap overlay + assets + infrastructure + river, click-to-query), **Satellite Change Detection** (pre/post NDWI, real Sentinel-2), **Model Performance** (CV metrics, feature importance, error analysis), **Client Data** (portfolio table + live ingestion uploader), **AI Analyst** (natural-language query → markdown report + downloadable + tool-call log).

## 13. API

FastAPI, `src/api/main.py`: `/health`, `/region`, `/predict`, `/assets/exposed`, `/model/metrics`, `/grid`, `/agent/analyze`, `/ingest`. Pydantic request/response models, a timing middleware that logs and headers every request's latency, and full OpenAPI docs at `/docs`.

## 14. Setup

```bash
git clone https://github.com/athish-shubhan/Inundata.git && cd Inundata
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add ANTHROPIC_API_KEY to enable the real LLM agent path
```

Data is already fetched and committed under `data/raw/` and `data/processed/` so the app runs immediately. To regenerate from scratch (hits public APIs, ~2 minutes):

```bash
python scripts/fetch_data.py       # SRTM elevation, Open-Meteo rainfall, GIS, client data
python scripts/fetch_satellite.py  # Sentinel-2 bands, clipped to the AOI
python scripts/run_pipeline.py     # fusion → train → evaluate
```

## 15. Running instructions

```bash
uvicorn src.api.main:app --reload --port 8000      # API + docs at http://localhost:8000/docs
streamlit run dashboard/app.py                       # dashboard at http://localhost:8501
```

Or via Docker:

```bash
docker compose up --build
```

## 16. Example workflow

1. Open the dashboard, **Risk Map** tab — see 39 of 480 grid cells at risk-score ≥ 0.5, concentrated near the river.
2. Click a point near downtown Hitoyoshi.
3. Go to **AI Analyst**, the click pre-fills a query; hit **Run analysis**.
4. Get a markdown report: risk band, driving features (elevation, HAND, distance-to-river, rainfall), top 5 exposed assets by value and risk, model caveats, recommended actions — downloadable, and reproducible via the same call to `POST /agent/analyze`.

## 17. Results (this repo's own last run — not fabricated)

- Pipeline (fusion → train → evaluate) completes in **~4.6s** on a 480-cell grid.
- Selected model: **random_forest**, spatial-CV F1 = 0.32, ROC-AUC = 0.93 (see caveat in §9).
- **35/35 pytest tests pass** across ingestion, geospatial, ML, agent, and API layers (`python -m pytest -q`).
- API, agent (deterministic-fallback mode), and dashboard were all exercised live during development — see §18 for what was and wasn't verified.

## 18. Verification performed

- ✅ `scripts/fetch_data.py` and `scripts/fetch_satellite.py` run against real public APIs and produced the committed `data/raw/*`.
- ✅ `scripts/run_pipeline.py` runs end-to-end and produced the committed `data/processed/*`.
- ✅ `pytest` — 35/35 passing.
- ✅ FastAPI server started and every endpoint (including `/ingest` with a real CSV and an error case) was hit with `curl` and returned expected responses.
- ✅ Streamlit dashboard was opened in a real browser; all five tabs, including running a live agent query, were clicked through and screenshotted.
- ⚠️ **LLM tool-calling path** (`llm_report` in `src/agents/agent.py`) is implemented against the documented Anthropic Messages API tool-use format but was **not exercised live** in this environment (no `ANTHROPIC_API_KEY` was configured). The deterministic fallback — which shares the same tools and produces the same numbers — was fully verified instead. Set `ANTHROPIC_API_KEY` and re-run to exercise the real LLM path.
- ⚠️ **Docker build was attempted and did not complete in this environment.** It hung indefinitely (zero CPU progress for 20+ minutes) on this machine's `colima`-based Docker backend and was killed rather than left running; this looks like a local VM/networking issue with this particular Docker setup, not a `Dockerfile` correctness issue, but it was **not verified end-to-end**. If you hit the same thing, try `docker compose build --no-cache` on a native Docker Desktop / Linux host, or run the app directly via the venv instructions in §14–15, which were fully verified.

## 19. Limitations

- Grid resolution (~650m) is coarse for parcel-level decisions — appropriate for regional triage only.
- ML label is a remote-sensing proxy, not verified flood-extent ground truth (see §9, §7).
- River geometry is hand-digitized and approximate, not OSM-sourced (Overpass unreachable in this sandbox).
- Client and infrastructure data are simulated.
- Single-AOI PoC: the deterministic agent NLU (`PLACES` dict) only knows this AOI's landmarks; the real LLM path generalizes better since Claude parses free text itself.
- Small, imbalanced training set (21 positive / 480 total) — metrics should be read directionally.

## 20. Future work

- Add Sentinel-1 SAR for cloud-penetrating flood-extent detection during the event itself.
- Swap the hand-digitized river line for an OSM/Overpass extract once network access allows it.
- Multi-AOI support: parameterize the AOI instead of hardcoding Hitoyoshi in `src/config.py`.
- Replace the flat-file model store with a proper model registry for multi-region deployment.
- Add a real historical flood-extent inventory (e.g., a purchased or government-provided dataset) to replace the NDWI proxy label.

---

## Technical highlights

| Capability | Evidence in this repo |
|---|---|
| Python | Entire codebase; type-hinted, tested |
| Data analytics / ML | `src/ml/` — two models compared, spatial CV, feature importance, error analysis |
| Geospatial data processing | `src/geo/` — CRS reprojection, raster resampling, zonal stats, distance calc |
| Remote sensing / satellite data | Real Sentinel-2 NDWI/NDVI + change detection (`src/geo/indices.py`) |
| Weather data | Real Open-Meteo historical rainfall, IDW-interpolated (`src/geo/weather.py`) |
| GIS data | River centerline + infrastructure GeoJSON, `src/ingestion/geojson_source.py` |
| Multimodal data fusion | `src/geo/fusion.py` — 5 modalities → 1 feature table |
| Spatial AI | HAND, distance-to-river, neighborhood features, spatial CV (`src/ml/features.py`) |
| LLM / agentic workflows | `src/agents/agent.py` — real Claude tool-calling loop + verified deterministic fallback |
| Automated ingestion, heterogeneous client data | `src/ingestion/` — CSV/GeoJSON/GeoTIFF, validation, `client_adapter.py` registry |
| FastAPI | `src/api/main.py` — 8 endpoints, Pydantic, OpenAPI |
| Dashboard / rapid prototyping | `dashboard/app.py` — 5-tab Streamlit app, verified live in-browser |
| Client-facing communication | Agent's markdown report; this README's §2–3 |
| Technical documentation | `docs/architecture.md`, `docs/data_sources.md`, `docs/design_notes.md` |
| Independent problem definition | AOI, use case, label design, and model choice were all made without external input — documented with reasoning throughout |

Built by Athish Shubhan — see `docs/design_notes.md` for the engineering rationale behind each design decision.

## License

MIT — see [`LICENSE`](LICENSE).
