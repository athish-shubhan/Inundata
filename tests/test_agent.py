from src.agents.agent import DEFAULT_PLACE, PLACES, deterministic_report, parse_query
from src.agents.tools import DISPATCH, TOOL_SPECS, call_tool, haversine_km


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
    for section in ("# Flood Risk Assessment", "## Event Context", "## Exposed Critical Infrastructure",
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
