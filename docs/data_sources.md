# Data Sources

AOI: Hitoyoshi basin and the Kuma River valley upstream through Kuma village, Kumamoto Prefecture, Japan (`LON 130.67–130.83, LAT 32.18–32.26`, configured in `config/hitoyoshi.json`). The Kuma River flooded here on 2020-07-04 during the July 2020 Kyushu heavy-rain disaster.

| Modality | Source | Access | License / terms | Resolution | Temporal coverage | Role |
|---|---|---|---|---|---|---|
| Elevation | [Copernicus GLO-30 DSM](https://registry.opendata.aws/copernicus-dem/) tile N32 E130 | Public COG on S3, windowed read, AOI + 0.03° margin | Copernicus DEM licence (free, attribution) | 1 arc-second (~30 m), bilinear to the 30 m grid | Static (TanDEM-X, 2011–2015) | Predictors: elevation, slope, TPI, flow accumulation, HAND |
| Satellite imagery | Sentinel-2 L2A, `sentinel-cogs` public bucket, tile 52SFA | Public COGs, windowed reads, no key | Copernicus open data | 10–20 m, area-averaged to 30 m (SCL: nearest) | Pre 2020-05-11 (S2A), post 2020-08-14 (S2B) | Pre-event NDWI/NDVI/NDBI as predictors; post-event only as a diagnostic |
| Flood extent (label) | [GSI](https://www.gsi.go.jp/) inundation-estimate map, Kuma River, issued 2020-07-04 20:00 JST (`20200703oame_kumagawa_0704dansaizu`) | Public XYZ tiles, zoom 15, mosaicked into a 0/1 mask | GSI terms of use (attribution) | ~4 m pixels at this latitude (z15), averaged to a flooded fraction per 30 m cell | Event peak, 2020-07-04 | Label: a cell is flooded if ≥ 50 % of it is inside the extent |
| River | OpenStreetMap, `waterway=river` named 球磨川 (26 ways) | Overpass API | ODbL | Vector, rasterised (all-touched) | Current OSM | Burned into the DEM for routing; distance-to-river predictor |
| Critical infrastructure | OpenStreetMap `amenity` ∈ {hospital, clinic, school, kindergarten, fire_station, police, townhall} (66 sites) | Overpass API | ODbL | Points (way/relation centroids) | Current OSM | Exposure reporting |
| Client portfolio | 120 simulated assets on randomly sampled real OSM building centroids (seed 42) | Overpass API for locations | Locations ODbL; attributes synthetic | Points | n/a | Exposure reporting; values, year built and occupants are simulated |
| Rainfall | [Open-Meteo Historical Archive](https://open-meteo.com/en/docs/historical-weather-api) | Public REST API, no key | CC-BY 4.0 (ERA5-based reanalysis) | ~10 km reanalysis cell, 3 sample points | Daily, 2020-07-01 to 05 vs 2020-06-01 to 05 | Context in reports only |

## CRS handling

- The analysis grid is UTM zone 52N (`EPSG:32652`), Sentinel-2's native CRS for this tile, so the 10 m bands are averaged rather than rotated. All distances, slopes and areas are computed in metres on this grid.
- The DEM (geographic, `EPSG:4326`) and the GSI mask (Web Mercator, `EPSG:3857`) are reprojected on read with `rasterio.warp.reproject`.
- Vector data (river, facilities, portfolio) stays in WGS84 and is transformed to grid row/column on lookup (`Grid.cell_of`).
- Web maps get rasters warped back to WGS84 (`Grid.to_wgs84`).

## Known limitations

1. **The label is an estimate.** GSI's rapid inundation map estimates water levels at the flood edge from photographs and spreads them over an elevation model; it is the best public extent for this event but not a surveyed boundary, and it covers the main-stem flood rather than every tributary or pluvial ponding.
2. **Surface, not terrain, model.** Copernicus GLO-30 includes buildings and canopy, which raises apparent elevation and HAND in built-up and forested cells.
3. **Optical blind spot.** Sentinel-2 cannot see through the storm clouds at the flood peak; the nearest clear scenes are about eight weeks before and six weeks after. That is why post-event imagery is a diagnostic here and not the label.
4. **Rainfall is coarse.** The reanalysis returns the same totals for all three sample points across the AOI, so it is reported as context and not used as a spatial predictor.
5. **Simulated client attributes.** Portfolio values, year built and occupants come from a seeded RNG; only the building locations are real.
