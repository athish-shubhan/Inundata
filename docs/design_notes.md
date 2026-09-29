# Design Notes

This project explores whether real public satellite, weather, and terrain data can be fused into a usable flood-risk tool for a real disaster region. The client engagement described in the README is simulated to give the system an end-to-end consumer; the elevation, rainfall, and satellite imagery underneath it are real public data for a real event (the July 2020 Kuma River flood in Hitoyoshi, Japan).

## Summary

A flood-risk assessment platform for the Kuma River basin in Japan, the site of a major 2020 flood disaster. It fuses real SRTM elevation, real historical rainfall, and real Sentinel-2 satellite imagery with a simulated client property portfolio, trains a spatially-validated ML model, and lets a non-technical user ask a plain-English question like "assess flood risk after recent rainfall" and get back a client-facing report, through an agent that only narrates and never calculates, so every number is traceable to a deterministic tool call.

## Overview

Start from the underlying problem: an insurer wants to triage flood exposure fast, without a GIS team. A real, well-documented flood event was chosen so the pipeline could be validated against reality instead of inventing numbers. Five data modalities (terrain, weather, satellite, GIS vector data, and client data) get fused onto one spatial grid. A Random Forest, chosen after comparing against logistic regression with *spatial* cross-validation (not random splits, because neighboring grid cells are correlated), scores flood susceptibility. That model backs three interfaces that all share the same code: a FastAPI service, a Streamlit dashboard, and an LLM agent that calls read-only tools and narrates the results, with a deterministic fallback so the whole thing runs without any API key.

## Technical walkthrough

AOI selection and why (real disaster, real data, tractable scope) → ingestion layer (`src/ingestion`, format-agnostic loaders + validation reports, a small registry so a new client format doesn't require new pipeline code) → geospatial fusion (`src/geo/fusion.py`: CRS reprojection between WGS84/UTM/a metric CRS, windowed COG reads of Sentinel-2 straight from the public S3 bucket instead of downloading full tiles, HAND and river-distance as terrain-based flood proxies, IDW rainfall interpolation, 3×3 neighborhood features for spatial context) → ML (`src/ml`: proxy label from real NDWI change detection, spatial GroupKFold CV, feature importance, honest small-sample/imbalance caveats surfaced everywhere the model's output appears) → agent (`src/agents`: four whitelisted read-only tools, a real Claude tool-calling loop, and a deterministic fallback that both share the same tool implementations) → API/dashboard as two thin, consistent front ends over the same model.

## Architecture rationale

See `docs/architecture.md` for the diagram. The key idea: three surfaces (API, dashboard, agent) all import the same `RiskModel` and the same `src/agents/tools.py` functions, so there is exactly one source of truth for any number the system produces. The dashboard and the agent cannot disagree.

## Hardest problem solved

Getting genuine before/after satellite change detection without a paid API or credentials. The public Earth Search STAC API was used to search Sentinel-2 L2A metadata, picking the lowest-cloud scenes bracketing the flood, then doing windowed reads directly off the public `sentinel-cogs` S3 bucket with `rasterio` (no full-tile download, no auth). The catch: optical imagery can't see through the storm clouds present *during* the actual flood peak, so the "before/after" comparison is really "before vs. seven weeks after," documented explicitly rather than implying the model sees the flood itself.

## Key trade-off

Using an NDWI-change proxy label instead of either (a) fabricating flood-extent ground truth, or (b) not building an ML component at all. The choice was to build something real and be explicit about what it can and can't claim, rather than either extreme. The honest cost: reported metrics (F1=0.32 for the selected model) look modest next to what a marketing-oriented demo might show; that's the right trade for a technical audience.

## ML approach

Two models, spatial cross-validation, and metrics chosen for an imbalanced classification problem (precision/recall/F1/ROC-AUC, not accuracy: 96% accuracy is trivial here since only 4.4% of cells are positive). Random forest won on F1 and ROC-AUC. In-sample error analysis is run separately from the CV metrics and both are labeled clearly, since conflating them is a common way to accidentally overstate model quality.

## Geospatial approach

CRS discipline (WGS84 for storage/interop, native UTM for satellite reads, a metric CRS for distance math) and using `rasterio.warp.reproject` to resample rasters of different native resolutions (SRTM ~90m, Sentinel-2 10–20m, the analysis grid ~650m) onto one common grid without ever reprojecting the same layer twice.

## Multimodal fusion approach

Each modality lands on the same analysis grid through a different mechanism appropriate to its type: rasters (elevation, satellite) via `reproject`, point weather stations via IDW interpolation, vector geometry (river) via point-to-line distance. The fusion step documents each layer's native resolution and how it was aligned, in `docs/data_sources.md` and inline in `src/geo/fusion.py`.

## Agent design

The agent is a tool-calling loop, not a chatbot bolted onto the project. It has exactly four tools, all read-only and all backed by the same deterministic pipeline the API uses. The loop is capped at 8 tool calls and the tool dispatch table is whitelisted as a basic safeguard. When there's no API key, a deterministic template narrator produces the identical structure from the identical tool calls. This was built specifically so the project is fully verifiable without requiring anyone to hand over a paid API key.

## Ingestion / handling messy client data

`src/ingestion` returns a structured `IngestReport` (errors vs. warnings vs. metadata) instead of raising on the first problem, so a caller can see *everything* wrong with a file at once: missing columns, duplicate IDs, invalid/null geometry (auto-repaired with `buffer(0)` where possible), and CRS mismatches (auto-reprojected). Adding a new client format means writing one small loader function and registering it in `client_adapter.LOADERS`. CSV, GeoJSON/Shapefile, and GeoTIFF all work through the same registry today.

## Limitations

- Proxy ML label, not verified flood-extent ground truth.
- Hand-digitized river line, not OSM-sourced (Overpass unreachable in this environment).
- ~650m grid: regional triage, not parcel-level.
- Simulated client/infrastructure data.
- LLM tool-calling path implemented but not live-tested without an API key in this environment (deterministic fallback was fully verified instead).

## Future improvements

SAR-based flood detection, multi-AOI support, a real flood-extent training label, model registry for scale; see README §20.

## Frequently asked questions

**Why random forest over logistic regression?** Spatial-CV F1 and ROC-AUC were both better, and tree ensembles capture the nonlinear terrain-rainfall interactions expected in flood susceptibility, without hand-engineering interaction terms.

**Why not just use accuracy?** The label is 4.4% positive; a model predicting "no flood risk" everywhere gets ~96% accuracy and is useless. Precision/recall/F1/ROC-AUC are the honest metrics for this imbalance.

**How do you know the CV isn't leaking?** `GroupKFold` over coarse spatial blocks (not `KFold`/random split) is used specifically because adjacent grid cells share almost identical terrain/rainfall features; a random split would put near-duplicate cells in train and test.

**What would break first at national scale?** The in-memory `features.csv` + single-process `RiskModel` in the API (see `docs/architecture.md`, Scalability). A real deployment would need tiled raster storage and a proper model-serving layer.

**How do you stop the LLM from hallucinating a risk score?** It can't produce one: the tool schema only returns numbers from deterministic function calls, and the system prompt explicitly instructs the model to never compute statistics itself. A production version would add an automated check that flags any numeric token in the final report not traceable to a tool result.

**Why Hitoyoshi specifically?** A real, well-documented, moderately-scoped flood event with genuinely fetchable public data (SRTM, Open-Meteo, Sentinel-2). That lets the pipeline be validated against something real instead of a synthetic AOI, without needing a paid data source.
