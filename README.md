# Inundata

An independent geospatial flood-intelligence platform. It takes a real disaster (the July 2020 Kuma River flood in Hitoyoshi, Kumamoto, Japan), fuses satellite, terrain, hydrology, weather, and GIS data on a 30 m grid, trains a flood-susceptibility model against the officially mapped flood extent, and exposes it through an agentic natural-language interface, a REST API, and a client-facing dashboard.

> **The client engagement described below is simulated**, built to work through the full problem end to end. Elevation, satellite imagery, rainfall, the river network, critical infrastructure, and the flood extent used as ground truth are all real public data for a real event. Only the client portfolio's attributes (values, year built, occupants) are synthetic; its locations are real building footprints.

---

## 1. Problem

A regional insurer/infrastructure operator in Kumamoto Prefecture wants to know, after a heavy rainfall event, which of its properties and which critical infrastructure sit in flood-prone terrain along the Kuma River, quickly, and in language a risk manager (not a GIS analyst) can act on.

## 2. Client (simulated)

**Client:** a mid-size property & casualty insurer with a portfolio of 120 residential/commercial/industrial/agricultural assets in the Hitoyoshi basin.
**Business problem:** rapidly triage flood exposure after a rainfall event, without a dedicated GIS team.
**Client data:** a CSV asset register (`data/hitoyoshi/raw/client/client_portfolio.csv`): lat/lon, building type, insured value, year built, occupants.
**External data:** Copernicus 30 m elevation, Sentinel-2 imagery, the OpenStreetMap river network and critical facilities, Open-Meteo rainfall, and the GSI inundation estimate for 2020-07-04.
**Question:** *"Assess the flood risk around this region after recent rainfall and identify exposed infrastructure."*
**Deliverable:** a client-facing markdown risk report, an interactive dashboard, and an API other systems can call.
**Success criteria:** an analyst can go from a plain-English question to a ranked, explainable list of at-risk assets and facilities, backed by a model whose skill has been measured honestly against what actually flooded.

## 3. Solution

```
fetch (public COGs/APIs) → 30 m grid fusion + DEM hydrology → spatially validated, calibrated model → risk raster → agent / API / dashboard
```

Every input is warped onto one 30 m UTM grid (508 × 305 cells, 15.2 × 9.2 km). Terrain, D8 flow routing and Height Above Nearest Drainage are computed from the DEM; pre-event Sentinel-2 adds land cover. Four models are compared with spatial cross-validation against the Geospatial Information Authority of Japan (GSI) flood map for 2020-07-04; the winner's cross-fitted, calibrated scores are written to `risk.tif`, so the serving layer is a raster lookup and every served score comes from a model that never saw that cell's label. An LLM-capable agent (with a fully functional deterministic fallback) turns a plain-English request into a structured, honest report by calling read-only tools; it never computes a number itself.

## 4. Architecture

See [`docs/architecture.md`](docs/architecture.md) for the component diagram, data flow, and trade-offs. Short version: `scripts/fetch_data.py` → `src/geo` (grid, terrain, hydrology, indices, fusion) → `src/ml` (features, train, evaluate, model) → `src/agents` / `src/api` → `dashboard/`, all reading the same `features.tif` / `risk.tif` so the three surfaces never disagree.

The study area is configuration, not code: `config/hitoyoshi.json` holds the bounding box, CRS, resolution, event dates, scene URLs and named places. Point `INUNDATA_AOI` at another file with the same fields to run elsewhere; outputs go to `data/<aoi id>/`.

## 5. Data

See [`docs/data_sources.md`](docs/data_sources.md) for the full source/license/resolution/limitations table. Summary:

- **Elevation:** Copernicus GLO-30 DSM, read as a window straight from the public COG bucket, with a 3 km margin so flow routing is not cut at the AOI edge.
- **Satellite:** Sentinel-2 L2A, pre-event 2020-05-11 and post-event 2020-08-14 (tile 52SFA), bands B03/B04/B08/B11 + SCL, windowed COG reads from `sentinel-cogs`.
- **Flood reference (label):** GSI rapid inundation-estimate map for the Kuma River, issued 2020-07-04, mosaicked from its zoom-15 tiles into a georeferenced mask.
- **GIS:** the Kuma River (球磨川, 26 OSM ways) and 66 critical facilities (hospitals, clinics, schools, kindergartens, fire stations, police, town hall), both from OpenStreetMap via Overpass.
- **Weather:** Open-Meteo historical archive (ERA5-based reanalysis), event window 2020-07-01 to 05 against 2020-06-01 to 05.
- **Client:** 120 simulated assets placed on real OSM building footprints.

`python scripts/fetch_data.py` downloads all of it; every step is idempotent and retries with backoff.

## 6. Geospatial processing

- One analysis grid (`src/geo/grid.py`) in UTM 52N (`EPSG:32652`); every raster is warped onto it with the resampling that fits the data: bilinear for elevation, area-average for 10 m reflectance and for the flood mask (so a cell carries its flooded fraction), nearest for the SCL class band.
- DEM hydrology in pure numpy (`src/geo/hydrology.py`): Priority-Flood depression filling (Barnes et al. 2014), D8 flow directions, flow accumulation, and HAND (Nobre et al. 2011) traced along each cell's own flow path. The OSM river is burned into the DEM so routing follows the real channel, and any cell draining more than 1 km² also counts as drainage.
- Distance to the mapped river via a Euclidean distance transform on the rasterised channel, topographic position index over a 300 m window, slope from metric gradients.
- Feature layers are written as one multi-band GeoTIFF with named bands (`features.tif`), so the whole stack opens in QGIS.

## 7. Remote sensing

NDWI, NDVI and NDBI from pre-event Sentinel-2 reflectance are model inputs; SCL cloud masking removes cloudy cells and pre-event open water (NDWI > 0) is treated as river channel, not flood. Post-event imagery is kept only as a diagnostic: v1 of this project used NDWI change as its label, and v2 measures how good that was (§9).

## 8. Multimodal fusion

`src/geo/fusion.py` builds `features.tif` with three kinds of band: **predictors**, all knowable before the event (elevation, slope, HAND, river distance, TPI, log flow accumulation, pre-event NDVI/NDWI/NDBI); the **label** (GSI flooded fraction and the ≥ 50 % flooded flag); and **diagnostics** (permanent water, cloud, NDWI change, the old proxy label). A test (`tests/test_ml.py`) asserts that no post-event or label band is ever a predictor. Of 154,940 cells, 153,491 are scored; 741 open-water and 768 cloudy cells are excluded.

Rainfall is deliberately not a feature: the reanalysis is on a ~10 km grid, so across a 15 km AOI it is effectively constant (all three sample points return the same 298 mm) and would only let a model memorise location. It is reported as event context instead.

## 9. Machine learning

Four candidates, each scored with **spatial** cross-validation: `StratifiedGroupKFold` over 2 km blocks (40 blocks, 5 folds), with the decision threshold tuned on an inner split of the training blocks only. The same protocol is repeated with a random `KFold` to show how much ordinary CV overstates skill on autocorrelated rasters. 7,680 of 153,491 scored cells (5.0 %) flooded.

| model | spatial PR-AUC | spatial CSI | spatial F1 | random-CV PR-AUC |
|---|---|---|---|---|
| hand_threshold (terrain-only baseline) | 0.101 ± 0.043 | 0.098 | 0.177 | 0.110 |
| logistic_regression | 0.629 ± 0.084 | 0.464 | 0.632 | 0.634 |
| **random_forest (selected)** | **0.820 ± 0.037** | **0.523** | **0.683** | 0.916 |
| hist_gradient_boosting | 0.813 ± 0.056 | 0.467 | 0.634 | 0.924 |

*(means ± std over 5 spatial folds; CSI is the critical success index, IoU of the flooded class. Regenerate with `python scripts/run_pipeline.py`; numbers are from this repo's last run.)*

- **Selection rule:** highest mean spatial-CV PR-AUC. The random-CV column is about 0.1 higher for the tree models: that gap is the leakage a random split would have hidden.
- **Calibration:** out-of-fold scores are isotonically calibrated with cross-fitting (each fold mapped by a calibrator fit on the others). Brier score 0.0209 raw → 0.0195 calibrated, against 0.0475 for predicting the base rate everywhere. HIGH starts at the trained operating threshold (0.41).
- **What is served:** these cross-fitted scores, not the final model's. Scoring the grid with the final model looked better (pooled PR-AUC 0.92) only because it had seen 40k of these labels and their neighbours, and it pushed about nine times as many cells to ≥ 0.99. A test (`test_served_scores_do_not_replay_training_labels`) keeps the served map from scoring above the spatial-CV estimate. The final model in `model.joblib` is for locations outside this grid.
- **Out-of-fold errors** (`src/ml/evaluate.py`, every cell scored by a model that never saw its block): precision 0.68, recall 0.76 at the operating threshold; false negatives and positives are written to `error_analysis.csv` and mapped in the dashboard.
- **Was the old label any good?** No. NDWI change (post minus pre > 0.04) agrees with the GSI extent at CSI 0.13 (recall 0.44, precision 0.16), and a random forest trained on it scores PR-AUC 0.27 against GSI, versus 0.82 when trained on GSI. That measured gap is why v2 switched label.
- **What drives it:** permutation importance, measured on held-out spatial folds, is led by elevation (0.64) and distance to the main river (0.38); HAND, flow accumulation, TPI, slope and NDVI follow at 0.05 to 0.07. HAND alone is a weak predictor here (the baseline row) because it is measured to the nearest drainage including small tributaries, while the 2020 flood was the Kuma main stem backing up across the valley floor; see the HAND-band breakdown from `python -m src.ml.evaluate`. HAND measured only to the main river was tried and did not help (PR-AUC 0.808 vs 0.820): half the grid drains out of the AOI before reaching the mapped channel, so it is undefined there.

**Caveat, stated everywhere this model's output appears:** the label is the GSI rapid estimate, itself a model-assisted map rather than a surveyed extent, and the model is trained and evaluated on one event. Scores are calibrated for this event; treat them as relative susceptibility for other storms or places.

## 10. Spatial AI

Hydrologically derived features rather than raw coordinates (flow-path HAND, flow accumulation, topographic position, distance to the mapped channel), spatially blocked cross-validation with inner-fold threshold tuning, and cross-fitted calibration, so every reported figure comes from cells the model did not see.

## 11. Agentic workflow

`src/agents/agent.py` implements a Claude tool-calling loop (`llm_report`) over six read-only tools defined in `src/agents/tools.py` with JSON schemas: `get_region_info`, `assess_region`, `list_exposed_assets`, `list_exposed_infrastructure`, `get_event_rainfall`, `get_model_metrics`. A fully functional **deterministic fallback** (`deterministic_report`) runs automatically when `ANTHROPIC_API_KEY` is unset; this is how the repo runs out of the box and how it was verified. Both paths call the same tools and get identical numbers; only the phrasing differs. Every tool call is logged with latency (`tool_log`), the LLM loop is capped at 8 tool calls in total (counted per call, not per model turn; `tests/test_llm_loop.py` drives it with a scripted fake client), and the toolset is a fixed whitelist; it cannot execute shell commands or write files. API calls time out after 60 s; if the LLM call fails (outage, timeout, rejected key) the request is answered by the deterministic path instead, with `mode: deterministic_fallback_llm_error`.

Each scored location also reports whether that cell was inside the observed 2020 extent, so a report can say "HIGH risk, and flooded in 2020", and ranks its elevation, river distance and HAND against the rest of the valley ("lower than 93 % of scored cells") so a non-specialist can see why it scored that way. Named places come from the AOI config (`downtown`, `hitoyoshi station`, `city hall`, `castle`, `nishise`, `kuma village`, ...), radii can be given in km or m.

Example query: *"Assess the flood risk around downtown Hitoyoshi after recent rainfall and identify exposed infrastructure within 2km."* Answer (deterministic mode): HIGH (0.56); 53 client assets within 2 km (¥2.32 bn), 25 HIGH risk (¥1.12 bn); 37 critical facilities, 17 HIGH risk.

## 12. Dashboard

Streamlit, five tabs: **Risk Map** (folium, calibrated risk overlay warped to WGS84, optional observed 2020 extent, assets coloured by band, OSM infrastructure, river, click-to-query), **Satellite & Flood Reference** (pre-event NDWI, NDWI change, GSI extent, and the proxy-vs-reference agreement), **Model Performance** (spatial vs random CV for every model, permutation importance, reliability curve, out-of-fold error map), **Client Data** (scored portfolio + live ingestion uploader), **AI Analyst** (natural-language query → markdown report + download + tool-call log).

## 13. API

FastAPI, `src/api/main.py`: `/health`, `/region`, `/predict`, `/assets/exposed`, `/infrastructure/exposed`, `/rainfall`, `/model/metrics`, `/grid?step=`, `/agent/analyze`, `/ingest`. Pydantic request/response models, validated coordinates (WGS84 range, no NaN/inf: 422), radius (0 to 20 km, the same limit the agent's tools enforce) and grid step, 404 for points outside the AOI, `/ingest` validation reports that flag out-of-range (e.g. swapped) coordinates and drop non-numeric ones (uploads are streamed to disk, capped at 50 MB with a 413, and parsed off the event loop), CORS limited to `CORS_ORIGINS`, rasters loaded at startup, a timing middleware that logs and headers every request's latency, and OpenAPI docs at `/docs`.

## 14. Setup

```bash
git clone https://github.com/athish-shubhan/Inundata.git && cd Inundata
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add ANTHROPIC_API_KEY to enable the real LLM agent path
```

Raw inputs and processed outputs are committed under `data/hitoyoshi/` so the app runs immediately. To regenerate from scratch:

```bash
python scripts/fetch_data.py       # DEM, Sentinel-2, weather, OSM river/infrastructure/buildings, GSI flood tiles
python scripts/run_pipeline.py     # features.tif → CV + training → risk.tif, metrics.json → error analysis (1 to 2.5 min)
```

## 15. Running instructions

```bash
uvicorn src.api.main:app --reload --port 8000      # API + docs at http://localhost:8000/docs
streamlit run dashboard/app.py                       # dashboard at http://localhost:8501
python -m pytest -q                                  # test suite
python scripts/bench_api.py --lat 32.2158 --lon 130.7535   # in-process API latency
```

Or via Docker:

```bash
docker compose up --build
```

## 16. Example workflow

1. Open the dashboard's **Risk Map** tab: 36 of 120 client assets are HIGH risk (¥1.43 bn of ¥5.14 bn insured), and 23 sit inside the 2020 flood extent. Toggle the observed extent on to compare.
2. Click a point near downtown Hitoyoshi.
3. Go to **AI Analyst**; the click pre-fills a query. Hit **Run analysis**.
4. Get a markdown report: risk band and calibrated score, terrain drivers, whether the cell flooded in 2020, top exposed assets and facilities, event rainfall, model caveats, and recommended actions. It can be downloaded, and reproduced with `POST /agent/analyze`.

## 17. Results (this repo's own last run)

- Selected model: **random_forest**, spatial-CV PR-AUC 0.82 ± 0.04 and CSI 0.52 against the GSI extent (base rate 5 %), versus 0.10 for a HAND-only baseline.
- Hindcast on the portfolio (served scores are out-of-fold): 20 of the 23 assets that flooded in 2020 are flagged HIGH, with 16 further HIGH flags on assets that stayed dry.
- Pipeline: features in 7 s (including D8 routing over ~360k cells with the 3 km margin), training and evaluation in 70 to 140 s on an 8-core laptop, depending on load.
- **88/88 pytest tests pass**, including hydrology checks on synthetic DEMs with known answers, a no-leakage test on the feature list, and a check that the served map does not replay training labels. CI (`.github/workflows/tests.yml`) runs pyflakes and the suite on every push; the same steps pass in a fresh venv built from `requirements.txt`.
- API latency (in-process, p50, `benchmarks/api_latency_{before,after}.json`): `/predict` 18.1 → 2.1 ms, `/assets/exposed` 125 → 3.8 ms, `/agent/analyze` 129 → 10 ms, first request after startup 4,822 → 6 ms (the rasters now load at startup instead of on the first request). `/grid` is 34 → 31 ms while returning 1,500 blocks of 300 m instead of 480 coarse cells. Single-machine timings; expect run-to-run variation.

## 18. Verification performed

- ✅ `scripts/fetch_data.py` ran against the public endpoints and produced the committed `data/hitoyoshi/raw/*`.
- ✅ `scripts/run_pipeline.py` ran end to end and produced the committed `data/hitoyoshi/processed/*`; a rerun reproduces the same metrics.
- ✅ `pytest`: 88/88 passing, locally and in a clean venv installed from `requirements.txt`.
- ✅ Dashboard executed headlessly with Streamlit's `AppTest` (all tabs render, an agent query runs) without exceptions.
- ✅ Every API endpoint exercised through FastAPI's `TestClient` in the test suite and the latency benchmark.
- ⚠️ **LLM tool-calling path** (`llm_report`) is implemented against the Anthropic Messages API tool-use format but was **not exercised live** (no `ANTHROPIC_API_KEY` configured). The deterministic fallback, which shares the same tools and numbers, was verified instead.
- ⚠️ **Docker build was not verified end to end.** An earlier attempt hung on this machine's `colima` backend; run directly via the venv instructions in §14–15 if you hit the same issue.

## 19. Limitations

- One event, one valley: the model has not been tested on another storm or basin, and "susceptibility" here means "resembles what flooded on 2020-07-04".
- The label is the GSI rapid estimate, not a surveyed extent; it may miss tributary or pluvial flooding outside the mapped main-stem area.
- The DEM is a surface model (buildings and trees included), which biases HAND upward in built-up and forested cells.
- 30 m cells are suitable for neighbourhood triage, not parcel-level decisions.
- Because each 2 km block is scored by a different fold model, neighbouring cells across a block edge differ slightly more than within a block (mean 0.017 vs 0.010); visible as faint seams in the risk map at most.
- Client portfolio attributes are simulated (locations are real buildings).
- The deterministic agent only knows the places listed in the AOI config; the LLM path handles free text.

## 20. Future work

- Validate on a second event (e.g. other 2020 Kyushu basins) before calling the scores transferable.
- Add Sentinel-1 SAR for an independent, cloud-penetrating flood extent at the event peak.
- Use a bare-earth DTM (e.g. GSI 5 m/10 m DEM) for HAND, and extend the mapped river beyond the AOI so main-stem HAND is defined everywhere.
- Event-specific inputs (river gauge stage, radar rainfall) to turn susceptibility into an event forecast.
- Replace flat files with a model registry and tiled raster store for multiple AOIs.

---

## What changed from v1

v1 used a 480-cell, ~650 m grid built from SRTM point samples, a hand-digitised river line, a single-elevation "HAND", and NDWI change as the label. Its bounding box (32.74–32.86 N) also sat about 65 km north of Hitoyoshi, which is at 32.21 N. v2 corrects the AOI, moves to a 30 m grid with real flow-routing hydrology and OSM geometry, trains on the official flood extent, and measures how weak the old proxy label was (§9).

## Technical highlights

| Capability | Evidence in this repo |
|---|---|
| Python | Entire codebase; type-hinted, tested |
| Data analytics / ML | `src/ml/`: four models, spatial vs random CV, inner-fold thresholds, cross-fitted calibration, permutation importance, out-of-fold error analysis |
| Geospatial data processing | `src/geo/grid.py`, `terrain.py`: one metric grid, resampling chosen per layer, rasterisation, distance transforms |
| Hydrology | `src/geo/hydrology.py`: Priority-Flood, D8, flow accumulation, flow-path HAND, tested on synthetic DEMs |
| Remote sensing / satellite data | Windowed Sentinel-2 COG reads, NDWI/NDVI/NDBI, SCL cloud mask (`src/geo/indices.py`, `scripts/fetch_data.py`) |
| Ground truth | GSI inundation tiles mosaicked into a georeferenced mask (`scripts/fetch_data.py:fetch_flood_reference`) |
| GIS data | OSM river, facilities and buildings via Overpass; `src/ingestion/geojson_source.py` |
| Multimodal data fusion | `src/geo/fusion.py`: terrain, hydrology, imagery and label in one named-band stack |
| LLM / agentic workflows | `src/agents/agent.py`: Claude tool-calling loop plus a verified deterministic fallback |
| Automated ingestion, heterogeneous client data | `src/ingestion/`: CSV/GeoJSON/GeoTIFF, validation reports, `client_adapter.py` registry |
| FastAPI | `src/api/main.py`: 10 endpoints, Pydantic, OpenAPI, latency benchmark |
| Dashboard / rapid prototyping | `dashboard/app.py`: 5-tab Streamlit app |
| Technical documentation | `docs/architecture.md`, `docs/data_sources.md`, `docs/design_notes.md` |

Built by Athish Shubhan. See `docs/design_notes.md` for the reasoning behind each design decision.

## License

MIT. See [`LICENSE`](LICENSE).
