# Data Sources

AOI: Hitoyoshi, Kumamoto Prefecture, Japan — Kuma River basin (`LAT 32.74–32.86, LON 130.68–130.82`), site of the July 2020 Kyushu floods, one of the more severe flood disasters in recent Japanese history.

| Modality | Source | Access | License / terms | Resolution | Temporal coverage | Real or simulated |
|---|---|---|---|---|---|---|
| Elevation (terrain) | SRTM 90m via [OpenTopoData](https://www.opentopodata.org/) | Public REST API, no key | SRTM: public domain (NASA/USGS) | ~90m, resampled to a 0.006° (~650m) analysis grid | Static (2000 SRTM mission) | Real |
| Weather / rainfall | [Open-Meteo Historical Archive API](https://open-meteo.com/en/docs/historical-weather-api) | Public REST API, no key | CC-BY 4.0 (Open-Meteo, ECMWF ERA5 reanalysis) | Point series, 3 stations IDW-interpolated to grid | Daily, 2020-06-01 to 2020-07-10 | Real |
| Satellite imagery | Sentinel-2 L2A, [Earth Search STAC](https://earth-search.aws.element84.com/v1) / `sentinel-cogs` public S3 bucket | Public COGs, no key | Copernicus open data (free, no restrictions) | 10–20m native, resampled to analysis grid | 2020-05-11 (pre-flood) and 2020-08-29 (post-flood), both <1.5% cloud | Real |
| River geometry | Manually digitized centerline, Kuma River through Hitoyoshi | — | — | Simplified 9-vertex polyline | Static | Approximate — OSM/Overpass was unreachable from this environment; coordinates hand-traced from public map imagery, not survey-grade |
| Critical infrastructure | Generated register (hospitals, schools, bridges, evacuation centers, fire stations) | — | — | Point locations, seeded RNG | Static | Simulated, documented as such |
| Client property portfolio | Generated register (120 assets: residential/commercial/industrial/agricultural) | — | — | Point locations + attributes, seeded RNG | Static | Simulated client dataset, documented as such |

## CRS handling

- All source data is reprojected to `EPSG:4326` (WGS84) at ingestion.
- Metric distance operations (river proximity) use `EPSG:6690` (JGD2000 / Japan Plane Rectangular, appropriate for this AOI).
- Sentinel-2 tiles are natively `EPSG:32652` (UTM zone 52N) and are reprojected on read via `rasterio.warp.reproject`.

## Known limitations

1. **Optical satellite blind spot during the event itself.** Sentinel-2 is optical and cannot see through the storm clouds present during the flood's peak (2020-07-04). The pre/post NDWI comparison uses the closest low-cloud scenes available (11 May and 29 Aug 2020), which captures a longer-term moisture/land-cover signature rather than instantaneous flood extent. A production system would add Sentinel-1 SAR, which penetrates cloud cover.
2. **River centerline is approximate.** OpenStreetMap's Overpass API was not reachable from this sandbox; the centerline was hand-digitized from public map imagery rather than pulled from OSM, so it is a simplified approximation, not survey-grade hydrography.
3. **Analysis grid resolution (~650m)** is coarse relative to river width and individual buildings; it smooths local water-extent signal and is appropriate for regional triage, not parcel-level flood mapping.
4. **No verified historical flood-extent polygon** was reproducibly available without paid data. The ML label is a proxy — see `docs/architecture.md` and `README.md` for how this affects the ML section.
5. **Client and infrastructure datasets are simulated**, generated with a seeded RNG within the AOI bounding box, and explicitly labeled as such everywhere they appear (README, dashboard, agent output).
