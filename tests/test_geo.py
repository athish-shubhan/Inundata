import numpy as np
from src.geo.terrain import load_grid, slope_deg
from src.geo.indices import river_distance_m, height_above_drainage, ndwi, ndvi
from src.geo.weather import idw_grid
from src.geo.zonal import sample_grid

def test_load_grid_shape_and_range():
    lats, lons, elev = load_grid()
    assert elev.shape == (len(lats), len(lons))
    assert np.nanmax(elev) > np.nanmin(elev)
    assert np.nanmax(elev) < 3000

def test_slope_non_negative():
    lats, lons, elev = load_grid()
    s = slope_deg(lats, lons, elev)
    assert np.nanmin(s) >= 0

def test_river_distance_non_negative():
    lats, lons, _ = load_grid()
    d = river_distance_m(lats, lons)
    assert np.nanmin(d) >= 0
    assert d.shape == (len(lats), len(lons))

def test_hand_relative_to_elevation():
    lats, lons, elev = load_grid()
    rd = river_distance_m(lats, lons)
    hand = height_above_drainage(elev, rd)
    assert hand.shape == elev.shape

def test_ndwi_ndvi_bounded():
    lats, lons, _ = load_grid()
    w = ndwi("pre_flood_2020-05-11", lats, lons)
    v = ndvi("pre_flood_2020-05-11", lats, lons)
    assert np.nanmax(w) <= 1.01 and np.nanmin(w) >= -1.01
    assert np.nanmax(v) <= 1.01 and np.nanmin(v) >= -1.01

def test_idw_grid_positive_rainfall():
    lats, lons, _ = load_grid()
    g = idw_grid(lats, lons, "flood_event")
    assert np.all(g > 0)

def test_zonal_sample_nearest():
    lats, lons, elev = load_grid()
    v = sample_grid(lats, lons, elev, lats[0], lons[0], window=0)
    assert v == elev[0, 0]
