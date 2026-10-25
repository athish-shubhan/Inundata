# Design Notes

This project explores whether real public satellite, terrain, and GIS data can be fused into a usable flood-risk tool for a real disaster region, and whether its skill holds up when measured against what actually flooded. The client engagement described in the README is simulated to give the system an end-to-end consumer; the data underneath it is real public data for a real event (the July 2020 Kuma River flood in Hitoyoshi, Japan).

## Summary

A flood-risk assessment platform for the Kuma River basin. It fuses Copernicus 30 m elevation, DEM-derived hydrology, pre-event Sentinel-2 land cover and OpenStreetMap geometry on one 30 m grid, trains against the official GSI flood extent for 2020-07-04 with spatial cross-validation, and lets a non-technical user ask a plain-English question and get back a client-facing report. The agent only narrates; every number traces to a deterministic tool call.

## Overview

An insurer wants to triage flood exposure fast, without a GIS team. A real, well-documented event was chosen so the pipeline could be validated against reality instead of inventing numbers. v1 of the project used a coarse grid and a satellite-change proxy label; v2 rebuilt the geospatial core on a 30 m grid with flow-routing hydrology and switched to the GSI extent as the label, after measuring that the proxy agreed with it at CSI 0.13. A random forest, chosen from four candidates by spatial-CV PR-AUC, backs three interfaces that share the same code: a FastAPI service, a Streamlit dashboard, and an LLM agent with a deterministic fallback, so the whole thing runs without an API key.

## Technical walkthrough

AOI as config (`config/hitoyoshi.json`) → idempotent fetch of public COGs, Overpass and the GSI tile layer (`scripts/fetch_data.py`) → one analysis grid with per-layer resampling (`src/geo/grid.py`) → terrain and hydrology (`terrain.py`, `hydrology.py`: Priority-Flood, D8, flow accumulation, flow-path HAND with the OSM river burned in) → spectral indices and cloud masking (`indices.py`) → a named-band stack separating predictors, label and diagnostics (`fusion.py`) → spatial CV with inner-fold thresholds, cross-fitted calibration, proxy-label check and permutation importance (`src/ml/train.py`) → scores written back to the grid (`risk.tif`) → raster lookups for agent, API and dashboard (`src/ml/model.py`).

## Architecture rationale

See `docs/architecture.md`. The key idea: training writes calibrated scores for every cell, and all three surfaces read that raster through one `RiskModel`, so they cannot disagree and no model inference happens in the request path.

## Hardest problems solved

- **Getting ground truth without paid data.** GSI publishes its inundation estimate only as styled map tiles. `fetch_flood_reference` computes the tile range for the AOI, treats 404s as "no water", reads each tile's alpha channel as the mask, and georeferences the mosaic in Web Mercator so it can be area-averaged onto the analysis grid like any other raster.
- **HAND that follows water, not straight lines.** The v1 "HAND" subtracted one river elevation from every cell. v2 fills depressions, routes D8 flow on a DEM padded by 3 km, and takes each cell's height above the drainage cell its own flow path reaches. Tested on synthetic valleys with known answers.

## Serving honest scores

The first v2 build scored the grid with the final model. Its map agreed with the 2020 extent at pooled PR-AUC 0.92, ten points above the spatial-CV estimate, because the model had been fit on 40k of those cells and every other cell sits next to one. The served map is now the cross-fitted out-of-fold score, which matches the CV estimate and is actually calibrated, and a test fails if the served map ever scores above it again. The cost is faint seams at 2 km block edges, where neighbouring cells come from different fold models.

## Key trade-off

Training on one event's extent makes the scores calibrated and checkable for that event, but says nothing yet about other storms. The alternative, a physically based or multi-event model, needs data this project does not have. The choice was to measure skill properly on what exists and state the single-event limit everywhere the score appears.

## ML approach

Spatially blocked 5-fold CV (2 km blocks) for every reported figure, including the decision threshold, which is tuned on an inner split of the training blocks. Metrics fit a 5 %-positive problem: PR-AUC for ranking, CSI/F1 at the operating threshold, Brier for calibration. Random CV is reported alongside to show the leakage it would hide (PR-AUC 0.92 vs 0.82 for the random forest). A HAND-only baseline (PR-AUC 0.10) shows what the learned model adds over the classic terrain rule.

## Geospatial approach

One metric grid in the satellite's native UTM zone; area-average resampling for 10 m reflectance and for the flood mask (so cells carry flooded fractions, not one sampled pixel), bilinear for elevation, nearest for class bands; a routing buffer so catchments are not cut at the edge; distance transforms instead of point-to-line loops.

## Multimodal fusion approach

Each modality lands on the grid through the mechanism its type needs: rasters through `warp`, vector lines through rasterisation and a distance transform, the tile layer through mosaic + georeference + warp, point weather as report context. Bands are grouped by what they may be used for, and a test enforces that post-event and label bands are never predictors.

## Agent design

A tool-calling loop over six read-only tools, capped at 8 calls, with a whitelisted dispatch table. Tool results are JSON with explicit `is_error` flags. The deterministic narrator produces the same sections from the same calls. Each location carries both the model's band and whether the cell flooded in 2020, which lets a report separate "the model thinks this is exposed" from "this actually flooded". Drivers are also given as ranks within the valley (elevation, river distance, HAND), which a risk manager can read without knowing what HAND is.

## Ingestion / handling messy client data

`src/ingestion` returns a structured `IngestReport` (errors vs warnings vs metadata) instead of raising on the first problem, covering missing columns, duplicate IDs, invalid or null geometry (auto-repaired with `buffer(0)` where possible) and CRS mismatches (auto-reprojected). A new format is one loader registered in `client_adapter.LOADERS`.

## Limitations

- One event, one valley; transferability is untested.
- GSI rapid estimate as label, not a surveyed extent.
- Surface model DEM biases HAND in built-up and forested cells.
- Simulated client attributes on real building locations.
- LLM tool-calling path implemented but not live-tested without an API key (the deterministic fallback was fully verified instead).

## Future improvements

A second event for out-of-event validation, Sentinel-1 SAR extent at the peak, a bare-earth DTM, gauge/radar inputs for event forecasting, tiled storage for many AOIs; see README §20.

## Frequently asked questions

**Why random forest?** Highest spatial-CV PR-AUC (0.82 ± 0.04) of the four candidates; gradient boosting is close (0.81) with higher fold variance. Both beat logistic regression (0.63) because the relationship between elevation, distance to the channel and flooding is strongly non-linear along a valley.

**Why not accuracy?** 95 % of cells stay dry, so "never floods" is 95 % accurate and useless. PR-AUC, CSI and F1 focus on the flooded class.

**How do you know the CV isn't leaking?** Folds are 2 km spatial blocks, thresholds are tuned inside the training blocks, calibration is cross-fitted, and the random-CV number is shown next to it so the size of the effect is visible.

**Why isn't rainfall a feature?** The reanalysis cell is about 10 km across, so all three sample points in the AOI return the same totals; as a feature it would carry no spatial signal and could only leak location.

**What would break first at national scale?** Holding the full stacks in memory per worker and the single-threaded hydrology loops (see `docs/architecture.md`). Both have standard fixes: windowed COG reads and a compiled routing library.

**How do you stop the LLM from hallucinating a risk score?** It can't produce one from the tools' point of view: every number comes from a deterministic function, and the system prompt forbids computing statistics. A production version would add a check that flags any number in the final report not present in a tool result.

**Why Hitoyoshi specifically?** A real, well-documented flood with a public official extent, public 30 m elevation and clear Sentinel-2 scenes either side, at a scale that runs on a laptop.
