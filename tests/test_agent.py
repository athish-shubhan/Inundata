from src.agents.agent import parse_query, deterministic_report, PLACES
from src.agents.tools import call_tool, haversine_km

def test_parse_query_explicit_coords():
    p = parse_query("check risk at 32.80, 130.76 within 5km")
    assert p["lat"] == 32.80 and p["lon"] == 130.76 and p["radius_km"] == 5.0

def test_parse_query_named_place_default_radius():
    p = parse_query("assess flood risk upstream after rainfall")
    assert (p["lat"], p["lon"]) == PLACES["upstream"]
    assert p["radius_km"] == 2.0

def test_parse_query_fallback_to_center():
    p = parse_query("what is the situation here")
    assert (p["lat"], p["lon"]) == PLACES["city center"]

def test_deterministic_report_structure():
    r = deterministic_report("Assess flood risk downtown within 2km")
    assert r["mode"] == "deterministic_fallback"
    assert "# Flood Risk Assessment" in r["report_markdown"]
    assert len(r["tool_log"]) == 4
    assert all(t["ok"] for t in r["tool_log"])

def test_call_tool_unknown_tool_logs_error():
    log = []
    result = call_tool("delete_everything", {}, log)
    assert "error" in result
    assert log[0]["ok"] is False

def test_haversine_zero_distance():
    assert haversine_km(32.8, 130.75, 32.8, 130.75) == 0
