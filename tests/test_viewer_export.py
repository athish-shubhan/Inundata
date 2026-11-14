import json

import numpy as np

from scripts.export_viewer import (FLAG_FLOODED, FLAG_PERMANENT_WATER, HAND_STEP_M, OUT_DIR, RISK_NODATA,
                                   build_scene, confusion, pack_layers)


def unpack(blob, n):
    elev = np.frombuffer(blob, dtype="<u2", count=n)
    hand, risk, flags = (np.frombuffer(blob, dtype=np.uint8, count=n, offset=n * (2 + k)) for k in range(3))
    return elev, hand, risk, flags


def test_pack_layers_round_trip_within_quantisation():
    elev = np.array([[77.3, 120.04], [595.5, 80.0]])
    hand = np.array([[0.0, 3.1], [500.0, 0.26]])
    risk = np.array([[0.0, 1.0], [np.nan, 0.4135]])
    flooded = np.array([[1.0, 0.0], [0.0, 1.0]])
    water = np.array([[0.0, 1.0], [0.0, 1.0]])
    blob, offset = pack_layers(elev, hand, risk, flooded, water)
    assert len(blob) == 4 * 5 and offset == 77.0
    e, h, r, f = unpack(blob, 4)
    assert np.allclose(e / 10 + offset, elev.ravel(), atol=0.05)
    assert np.allclose(h[[0, 1, 3]] * HAND_STEP_M, hand.ravel()[[0, 1, 3]], atol=HAND_STEP_M / 2)
    assert h[2] == 255                      # heights beyond the range are capped, not wrapped
    assert r[2] == RISK_NODATA and r[1] == 254 and r[0] == 0
    assert abs(r[3] / 254 - 0.4135) < 1 / 254
    assert list(f) == [FLAG_FLOODED, FLAG_PERMANENT_WATER, 0, FLAG_FLOODED | FLAG_PERMANENT_WATER]


def test_confusion_counts_ignore_cells_without_a_prediction():
    risk = np.array([0.9, 0.9, 0.1, 0.1, np.nan])
    flooded = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert confusion(risk, flooded, 0.5) == {"hit": 1, "miss": 1, "false_alarm": 1, "correct_dry": 1}


def test_scene_is_consistent_with_the_grid():
    blob, scene = build_scene()
    n = scene["width"] * scene["height"]
    assert len(blob) == n * 5
    _, _, risk, flags = unpack(blob, n)
    assert sum(scene["confusion"].values()) == int((risk != RISK_NODATA).sum())
    assert scene["observed_flood_cells"] == int((flags & FLAG_FLOODED > 0).sum())
    assert 0 < scene["threshold"] < 1
    for item in scene["assets"] + scene["facilities"]:
        assert 0 <= item["col"] < scene["width"] and 0 <= item["row"] < scene["height"]
        assert item["risk"] is None or 0 <= item["risk"] <= 1
    assert len(scene["assets"]) > 100 and scene["river"]


def test_committed_viewer_data_matches_the_pipeline_outputs():
    # If risk.tif or features.tif change, the viewer must be re-exported (make viewer-data).
    blob, scene = build_scene()
    assert (OUT_DIR / "terrain.bin").read_bytes() == blob
    assert json.loads((OUT_DIR / "scene.json").read_text()) == json.loads(json.dumps(scene))
