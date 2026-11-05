import numpy as np
from affine import Affine

from src.config import DATA_PROC, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
from src.geo.grid import Grid, analysis_grid, read_stack, write_stack
from src.geo.hydrology import d8_downstream, fill_depressions, flow_accumulation, hand_from_dem, height_above_drainage
from src.geo.indices import normalized_difference
from src.geo.terrain import distance_to, slope_deg, topographic_position


def _valley(h=21, w=15, side_slope=1.0, down_slope=0.1):
    """V-shaped valley draining south: channel down the middle column."""
    rows, cols = np.mgrid[0:h, 0:w]
    return (np.abs(cols - w // 2) * side_slope + (h - rows) * down_slope).astype("float64")


def test_fill_depressions_removes_pits():
    dem = _valley()
    dem[10, 3] -= 5  # a closed pit on the valley side
    filled = fill_depressions(dem)
    assert np.all(filled >= dem)
    down = d8_downstream(filled, 30.0)
    interior = np.zeros(dem.shape, bool)
    interior[1:-1, 1:-1] = True
    assert np.all(down.reshape(dem.shape)[interior] >= 0), "every interior cell must drain somewhere"


def test_flow_accumulation_conserves_cells():
    dem = _valley()
    filled = fill_depressions(dem)
    acc = flow_accumulation(filled, d8_downstream(filled, 30.0))
    assert acc.min() >= 1
    outlet = acc[-1, dem.shape[1] // 2]
    assert outlet == acc.max()
    assert outlet > dem.size / 2  # most of the valley drains through the channel outlet


def test_hand_is_zero_on_drainage_and_uses_the_channel_a_cell_drains_to():
    dem = _valley()
    filled = fill_depressions(dem)
    down = d8_downstream(filled, 30.0)
    drainage = np.zeros(dem.shape, bool)
    drainage[:, dem.shape[1] // 2] = True
    hand = height_above_drainage(dem, filled, down, drainage)
    assert np.all(hand[drainage] == 0)
    # side_slope=1 m per cell, so a cell k columns from the channel sits about k m above it
    assert abs(hand[10, dem.shape[1] // 2 + 3] - 3) < 0.5
    assert np.all(np.diff(hand[10, dem.shape[1] // 2:]) > 0)


def test_hand_from_dem_marks_large_contributing_area_as_drainage():
    dem = _valley(h=41, w=31)
    river = np.zeros(dem.shape, bool)
    out = hand_from_dem(dem, 30.0, river, channel_area_m2=20 * 900)
    assert out["drainage"].any()
    assert np.nanmin(out["hand"]) == 0


def test_slope_of_known_plane():
    rows, cols = np.mgrid[0:10, 0:10]
    plane = cols * 30.0  # rises 30 m per 30 m cell
    assert np.allclose(slope_deg(plane.astype("float32"), 30.0), 45, atol=1e-4)


def test_topographic_position_negative_in_valley():
    tpi = topographic_position(_valley(h=41, w=41).astype("float32"), 30.0, radius_m=150)
    assert tpi[20, 20] < 0 < tpi[20, 2]


def test_distance_to_mask():
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    d = distance_to(mask, 30.0)
    assert d[2, 2] == 0 and d[2, 3] == 30 and np.isclose(d[3, 3], 30 * np.sqrt(2))


def test_normalized_difference_bounded_and_nan_safe():
    a = np.array([0.2, 0.0, 0.5], dtype="float32")
    b = np.array([0.1, 0.0, 0.5], dtype="float32")
    nd = normalized_difference(a, b)
    assert np.isclose(nd[0], 1 / 3) and np.isnan(nd[1]) and nd[2] == 0


def test_analysis_grid_covers_aoi_and_round_trips():
    g = analysis_grid()
    assert g.res == 30
    lon, lat = g.lonlat()
    assert lon.min() <= LON_MIN + 0.001 and lon.max() >= LON_MAX - 0.001
    assert lat.min() <= LAT_MIN + 0.001 and lat.max() >= LAT_MAX - 0.001
    r, c, inside = g.cell_of(lat[[10, 200], [20, 400]], lon[[10, 200], [20, 400]])
    assert inside.all() and list(r) == [10, 200] and list(c) == [20, 400]
    _, _, inside = g.cell_of(0.0, 0.0)
    assert not inside[0]


def test_buffered_grid_keeps_alignment():
    g = Grid(Affine(30, 0, 1000, 0, -30, 5000), 10, 20)
    b = g.buffered(3)
    assert b.shape == (16, 26)
    assert (b.transform.c, b.transform.f) == (910, 5090)


def test_stack_round_trip(tmp_path):
    g = Grid(Affine(30, 0, 657300, 0, -30, 3570750), 4, 5, "EPSG:32652")
    layers = {"a": np.arange(20, dtype="float32").reshape(4, 5), "b": np.full((4, 5), np.nan, "float32")}
    write_stack(tmp_path / "s.tif", g, layers, tags={"k": 1})
    g2, back, tags = read_stack(tmp_path / "s.tif")
    assert g2 == g and list(back) == ["a", "b"] and tags["k"] == "1"
    assert np.array_equal(back["a"], layers["a"]) and np.isnan(back["b"]).all()


def test_feature_stack_on_analysis_grid():
    g, layers, _ = read_stack(DATA_PROC / "features.tif")
    assert g == analysis_grid()
    for name in ("elev_m", "hand_m", "flooded", "permanent_water", "cloud", "proxy_label"):
        assert name in layers
    assert 0 < np.nanmean(layers["flooded"]) < 0.2
    assert np.nanmin(layers["hand_m"]) >= 0


def test_cell_of_treats_invalid_coordinates_as_outside():
    from src.ml.model import RiskModel
    grid = RiskModel().grid
    lat = np.array([32.2158, np.nan, 1000.0, 32.2, np.inf, 91.5])
    lon = np.array([130.7535, 130.75, 130.75, -500.0, 130.7, 130.7])
    r, c, inside = grid.cell_of(lat, lon)
    assert inside.tolist() == [True, False, False, False, False, False]
    assert (r[1:] == -1).all() and (c[1:] == -1).all()
