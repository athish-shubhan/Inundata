import pytest

from src.agents.agent import DEFAULT_PLACE, PLACES, deterministic_report, parse_query
from src.agents.tools import DISPATCH, MAX_RADIUS_KM, TOOL_SPECS, call_tool, haversine_km


def test_parse_query_explicit_coords():
    p = parse_query("check risk at 32.21, 130.75 within 5km")
    assert (p["lat"], p["lon"], p["radius_km"], p["place"]) == (32.21, 130.75, 5.0, None)


def test_parse_query_named_place_default_radius():
    p = parse_query("assess flood risk upstream after rainfall")
    assert (p["lat"], p["lon"]) == PLACES["upstream"]
    assert p["radius_km"] == 2.0


def test_parse_query_prefers_longest_place_name():
    assert parse_query("assets near kuma village")["place"] == "kuma village"


def test_parse_query_metres():
    assert parse_query("downtown within 500 m")["radius_km"] == 0.5


def test_parse_query_fallback_to_default_place():
    p = parse_query("what is the situation here")
    assert p["place"] == DEFAULT_PLACE and (p["lat"], p["lon"]) == PLACES[DEFAULT_PLACE]


def test_deterministic_report_structure():
    r = deterministic_report("Assess flood risk downtown within 2km")
    assert r["mode"] == "deterministic_fallback"
    md = r["report_markdown"]
    for section in ("# Flood Risk Assessment", "## Event Context", "In valley terms", "## Exposed Critical Infrastructure",
                    "## Confidence & Limitations"):
        assert section in md
    assert [t["tool"] for t in r["tool_log"]] == ["get_region_info", "assess_region", "list_exposed_assets",
                                                  "list_exposed_infrastructure", "get_event_rainfall", "get_model_metrics"]
    assert all(t["ok"] for t in r["tool_log"])


def test_deterministic_report_outside_aoi():
    r = deterministic_report("risk at 35.0, 139.0")
    assert "outside the analysis area" in r["report_markdown"]


def test_tool_specs_match_dispatch():
    assert [t["name"] for t in TOOL_SPECS] == list(DISPATCH)


def test_call_tool_unknown_tool_logs_error():
    log = []
    result = call_tool("delete_everything", {}, log)
    assert "error" in result
    assert log[0]["ok"] is False


def test_haversine():
    assert haversine_km(32.2, 130.75, 32.2, 130.75) == 0
    assert abs(haversine_km(32.0, 130.0, 33.0, 130.0) - 111.2) < 0.5


def test_parse_query_clamps_radius_to_supported_range():
    far = parse_query("downtown within 300 km")
    assert far["radius_km"] == MAX_RADIUS_KM and far["radius_clamped_from"] == 300
    tiny = parse_query("downtown within 0 km")
    assert tiny["radius_km"] == 0.05 and tiny["radius_clamped_from"] == 0
    assert parse_query("downtown within 2 km")["radius_clamped_from"] is None


def test_report_says_when_radius_was_clamped():
    md = deterministic_report("downtown within 300 km")["report_markdown"]
    assert "Requested radius 300 km is outside the supported" in md and f"Within {MAX_RADIUS_KM:g} km" in md


@pytest.mark.parametrize("radius", [-3, 0, 5000])
def test_tools_reject_out_of_range_radius(radius):
    log = []
    for name in ("list_exposed_assets", "list_exposed_infrastructure"):
        out = call_tool(name, {"lat": 32.2158, "lon": 130.7535, "radius_km": radius}, log)
        assert "radius_km must be" in out["error"]
    assert not any(t["ok"] for t in log)


def test_tool_spec_advertises_radius_limit():
    spec = next(t for t in TOOL_SPECS if t["name"] == "list_exposed_assets")
    assert spec["input_schema"]["properties"]["radius_km"]["maximum"] == MAX_RADIUS_KM


@pytest.mark.parametrize("query", ["risk at 91.5, 130.7", "risk at 32.2, 999.0"])
def test_report_for_impossible_coordinates_is_outside_area(query):
    md = deterministic_report(query)["report_markdown"]
    assert "outside the analysis area" in md and "PROJ" not in md
