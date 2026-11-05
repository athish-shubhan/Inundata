import json
import re
import time

from src.agents.llm import LLM_ERRORS, SYSTEM_PROMPT, client, has_llm
from src.agents.tools import DISPATCH, MAX_RADIUS_KM, TOOL_SPECS, call_tool
from src.config import AOI, LLM_MODEL
from src.services.logging import get_logger

logger = get_logger("agent")

MAX_TOOL_CALLS = 8
MIN_RADIUS_KM = 0.05  # below one or two grid cells a radius search is meaningless
PLACES = {name: tuple(ll) for name, ll in AOI["places"].items()}
DEFAULT_PLACE = AOI["default_place"]
FEATURE_LABELS = {"elev_m": "elevation", "slope_deg": "slope", "hand_m": "HAND", "river_dist_m": "distance to river",
                  "tpi_m": "topographic position", "log_flow_acc": "flow accumulation", "ndvi_pre": "vegetation (NDVI)",
                  "ndwi_pre": "surface wetness (NDWI)", "ndbi_pre": "built-up index (NDBI)"}


def parse_query(q: str) -> dict:
    ql = q.lower()
    place = None
    m = re.search(r"(-?\d{1,2}\.\d+)\s*[,/]\s*(-?\d{2,3}\.\d+)", q)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
    else:
        place = next((p for p in sorted(PLACES, key=len, reverse=True) if p in ql), DEFAULT_PLACE)
        lat, lon = PLACES[place]
    r = re.search(r"(\d+(?:\.\d+)?)\s*(km|kilomet(?:er|re)s?|m|met(?:er|re)s?)\b", ql)
    asked = 2.0 if not r else float(r.group(1)) / (1 if r.group(2).startswith("k") else 1000)
    radius = min(max(asked, MIN_RADIUS_KM), MAX_RADIUS_KM)
    return {"lat": lat, "lon": lon, "radius_km": radius, "place": place,
            "radius_clamped_from": None if radius == asked else asked}


def _score_text(risk: dict) -> str:
    if risk.get("risk_score") is None:
        reason = {"open_water": "open water before the event (river channel)", "cloud": "cloud-covered in the pre-event image"}
        return f"**not scored** ({reason.get(risk.get('status'), 'no data')})"
    return f"**{risk['risk_band']}** (calibrated score {risk['risk_score']:.2f})"


def deterministic_report(query: str, reason: str = "no LLM API key configured") -> dict:
    log = []
    region = call_tool("get_region_info", {}, log)
    p = parse_query(query)
    at = {"lat": p["lat"], "lon": p["lon"]}
    near = at | {"radius_km": p["radius_km"]}
    risk = call_tool("assess_region", at, log)
    assets = call_tool("list_exposed_assets", near, log)
    infra = call_tool("list_exposed_infrastructure", near, log)
    rain = call_tool("get_event_rainfall", {}, log)
    metrics = call_tool("get_model_metrics", {}, log)

    where = f"{p['place']} ({p['lat']:.4f}, {p['lon']:.4f})" if p["place"] else f"({p['lat']:.4f}, {p['lon']:.4f})"
    lines = [f"# Flood Risk Assessment: {region['name']}", "", f"**Query:** {query}", ""]
    if "error" in risk:
        lines += ["## Summary", f"No assessment possible at {where}: {risk['error']}."]
        return {"report_markdown": "\n".join(lines), "tool_log": log, "mode": "deterministic_fallback"}

    totals = max(rain["points"].values(), key=lambda v: v["event_total_mm"])
    d, pct = risk["drivers"], risk["driver_percentiles"]
    lines += [
        "## Summary",
        f"Flood susceptibility at {where} is {_score_text(risk)}. "
        f"Within {p['radius_km']:g} km: {assets['n_assets']} client assets (¥{assets['total_value_jpy']:,}), "
        f"{assets['n_high_risk']} HIGH risk (¥{assets['high_risk_value_jpy']:,}); "
        f"{infra['n_sites']} critical-infrastructure sites, {infra['n_high_risk']} HIGH risk."
        + (f" (Requested radius {p['radius_clamped_from']:g} km is outside the supported "
           f"{MIN_RADIUS_KM:g} to {MAX_RADIUS_KM:g} km, so {p['radius_km']:g} km was used.)"
           if p["radius_clamped_from"] is not None else ""), "",
        "## Event Context",
        f"- Rainfall {rain['window']}: up to {totals['event_total_mm']:.0f} mm, peaking at {totals['event_max_day_mm']:.0f} mm "
        f"on {totals['event_max_day']}, against {totals['baseline_total_mm']:.0f} mm over {rain['baseline_window']} "
        "(Open-Meteo reanalysis, about 10 km resolution).", "",
        "## Risk Assessment",
        f"- {risk['grid_cell']['size_m']:.0f} m grid cell centred at ({risk['grid_cell']['lat']:.4f}, {risk['grid_cell']['lon']:.4f})",
        f"- Elevation {d['elev_m']:.1f} m; height above nearest drainage (HAND) {d['hand_m']:.1f} m; "
        f"{d['river_dist_m']:.0f} m from the mapped river channel",
        f"- In valley terms: lower than {100 - pct['elev_m']:.0f}% of scored cells, closer to the river than "
        f"{100 - pct['river_dist_m']:.0f}%, and less HAND than {100 - pct['hand_m']:.0f}% "
        f"(strongest model drivers: {', '.join(FEATURE_LABELS.get(f, f) for f in metrics['top_features'][:3])})",
        f"- Topographic position {d['tpi_m']:+.1f} m relative to the surrounding 300 m (negative = valley floor)",
        f"- Observed on 2020-07-04: {'inside' if risk['observed_2020']['flooded'] else 'outside'} the GSI inundation "
        f"estimate ({risk['observed_2020']['flood_fraction']:.0%} of the cell)", "",
        "## Exposed Assets (top 5 by risk)",
    ]
    for a in assets["assets"][:5]:
        score = "" if a["risk_score"] is None else f" ({a['risk_score']:.2f})"
        lines.append(f"- {a['asset_id']} ({a['building_type']}, ¥{a['value_jpy']:,}, {a['occupants']} occupants), "
                     f"{a['dist_km']} km away: {a['risk_band']}{score}"
                     f"{', flooded in 2020' if a['flooded_2020'] else ''}")
    if not assets["assets"]:
        lines.append("- No client assets registered within the search radius.")
    lines += ["", "## Exposed Critical Infrastructure (top 5 by risk)"]
    for s in infra["sites"][:5]:
        name = f"unnamed {s['kind']}" if s["name"] == s["kind"] else f"{s['name']} ({s['kind']})"
        lines.append(f"- {name}, {s['dist_km']} km away: {s['risk_band']}"
                     f"{', flooded in 2020' if s['flooded_2020'] else ''}")
    if not infra["sites"]:
        lines.append("- No mapped critical infrastructure within the search radius.")
    sp, rnd = metrics["spatial_cv"], metrics["random_cv"]
    lines += [
        "", "## Confidence & Limitations",
        f"- Model: {metrics['selected_model']}, spatial cross-validation PR-AUC {sp['pr_auc']:.2f} and CSI {sp['csi']:.2f} "
        f"on held-out 2 km blocks (base rate {metrics['prevalence']:.0%}). A random split would report PR-AUC "
        f"{rnd['pr_auc']:.2f}, which overstates skill on neighbouring cells.",
        f"- {metrics['caveat']}",
        f"- Generated by the deterministic fallback narrator ({reason}); numbers are identical to the LLM path, only phrasing differs.",
        "", "## Recommended Actions",
        "- Field-verify HIGH-risk assets and sites before acting on this output.",
        "- Prioritise occupied buildings and critical infrastructure that are both HIGH risk and were flooded in 2020.",
    ]
    return {"report_markdown": "\n".join(lines), "tool_log": log, "mode": "deterministic_fallback"}


def llm_report(query: str) -> dict:
    """Claude tool-calling loop, capped at MAX_TOOL_CALLS tool executions in total.

    One model turn can request several tools at once, so the cap counts calls, not turns.
    Requests past the cap get an error result telling the model to finish; if it asks for
    tools again after that, the loop stops and returns a truncated result.
    """
    log = []
    c = client()
    messages = [{"role": "user", "content": query}]
    n_calls, refused = 0, False
    while True:
        resp = c.messages.create(model=LLM_MODEL, max_tokens=4000, system=SYSTEM_PROMPT, tools=TOOL_SPECS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            mode = "llm_agent" if resp.stop_reason == "end_turn" else f"llm_agent_{resp.stop_reason}"
            return {"report_markdown": text, "tool_log": log, "mode": mode}
        if refused:  # already told the budget was spent and still asking for tools
            return {"report_markdown": "agent exceeded tool-call budget without finishing.", "tool_log": log,
                    "mode": "llm_agent_truncated"}
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if n_calls >= MAX_TOOL_CALLS:
                refused = True
                out = {"error": f"tool-call budget of {MAX_TOOL_CALLS} spent; write the report from the results you have"}
            elif block.name not in DISPATCH:
                n_calls += 1
                out = {"error": "tool not permitted"}
            else:
                n_calls += 1
                out = call_tool(block.name, block.input, log)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": "error" in out,
                                 "content": json.dumps(out, ensure_ascii=False, default=float)})
        messages.append({"role": "user", "content": tool_results})


def run_agent(query: str) -> dict:
    t0 = time.perf_counter()
    if has_llm():
        try:
            result = llm_report(query)
        except LLM_ERRORS as e:  # outage, timeout, bad key: still answer, with the same numbers
            logger.warning(f"LLM call failed ({type(e).__name__}: {e}); using the deterministic report")
            result = deterministic_report(query, reason=f"LLM call failed: {type(e).__name__}")
            result["mode"] = "deterministic_fallback_llm_error"
    else:
        result = deterministic_report(query)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["query"] = query
    return result
