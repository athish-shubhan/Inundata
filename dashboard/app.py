import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import folium
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.agents.agent import run_agent
from src.agents.tools import get_event_rainfall, get_region_info
from src.config import DATA_PROC, DATA_RAW, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
from src.ingestion.client_adapter import ingest_client_dataset
from src.ml.model import RiskModel

st.set_page_config(page_title="Inundata", layout="wide")


@st.cache_resource
def get_model():
    return RiskModel()


@st.cache_data
def overlays():
    rm = get_model()
    risk, bounds = rm.grid.to_wgs84(rm.risk)
    flooded, _ = rm.grid.to_wgs84(np.where(rm.layers["flooded"] == 1, 1.0, np.nan))
    return risk, flooded, bounds


@st.cache_data
def scored_portfolio():
    return get_model().predict_points(pd.read_csv(DATA_RAW / "client" / "client_portfolio.csv"))


def array_to_rgba(arr, cmap, vmin, vmax, alpha=1.0):
    rgba = plt.get_cmap(cmap)(plt.Normalize(vmin=vmin, vmax=vmax)(np.nan_to_num(arr, nan=vmin)))
    rgba[..., 3] = alpha
    rgba[np.isnan(arr)] = [0, 0, 0, 0]
    return rgba


def raster_panel(col, arr, title, cmap, vmin, vmax, extent_km):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent_km, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("km east", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03)
    col.pyplot(fig)
    plt.close(fig)


rm = get_model()
region = get_region_info()
portfolio = scored_portfolio()
infra = json.loads((DATA_RAW / "gis" / "infrastructure.geojson").read_text())
river = json.loads((DATA_RAW / "gis" / "river.geojson").read_text())
metrics = json.loads((DATA_PROC / "metrics.json").read_text())
best = metrics["selected_model"]
extent_km = [0, rm.grid.width * rm.grid.res / 1000, 0, rm.grid.height * rm.grid.res / 1000]

st.title("Inundata: Kuma River Basin Flood Intelligence")
st.caption(f"Simulated client engagement · {region['name']} · {region['context']}")

tab_map, tab_sat, tab_model, tab_client, tab_agent = st.tabs(
    ["Risk Map", "Satellite & Flood Reference", "Model Performance", "Client Data", "AI Analyst"])

with tab_map:
    c1, c2 = st.columns([3, 1])
    high = portfolio[portfolio.risk_band == "HIGH"]
    with c2:
        st.metric("Client assets in AOI", len(portfolio))
        st.metric("Total exposure", f"¥{portfolio.value_jpy.sum():,.0f}")
        st.metric("HIGH-risk assets", f"{len(high)} (¥{high.value_jpy.sum():,.0f})")
        st.metric("Assets inside the 2020 flood extent", int(portfolio.flooded_2020.sum()))
        show_obs = st.checkbox("Show observed 2020 flood extent (GSI)", False)
        show_assets = st.checkbox("Show client assets", True)
        show_infra = st.checkbox("Show critical infrastructure", True)
        show_river = st.checkbox("Show river", True)
    with c1:
        risk_img, flooded_img, bounds = overlays()
        m = folium.Map(location=[(LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2], zoom_start=13, tiles="OpenStreetMap")
        folium.raster_layers.ImageOverlay(array_to_rgba(risk_img, "RdYlBu_r", 0, 1, 0.55), bounds=bounds,
                                          name="Flood susceptibility").add_to(m)
        if show_obs:
            folium.raster_layers.ImageOverlay(array_to_rgba(flooded_img, "Blues", 0, 1, 0.6), bounds=bounds,
                                              name="Observed 2020-07-04 (GSI)").add_to(m)
        if show_river:
            folium.GeoJson(river, name="Kuma River", style_function=lambda _: {"color": "#1f4e9c", "weight": 2}).add_to(m)
        if show_infra:
            for f in infra["features"]:
                lo, la = f["geometry"]["coordinates"]
                p = f["properties"]
                folium.CircleMarker([la, lo], radius=4, color="black", fill=True, fill_opacity=0.8,
                                    popup=f"{p['name']} ({p['kind']})").add_to(m)
        if show_assets:
            for _, r in portfolio.iterrows():
                folium.CircleMarker([r.lat, r.lon], radius=3, color="#d62728" if r.risk_band == "HIGH" else "#1f77b4",
                                    fill=True, fill_opacity=0.7,
                                    popup=f"{r.asset_id} ¥{r.value_jpy:,} · {r.risk_band}").add_to(m)
        folium.LayerControl().add_to(m)
        out = st_folium(m, height=560, width=None)
    st.caption(f"Colour = calibrated probability that a 30 m cell floods in an event like 2020-07-04 "
               f"(HIGH ≥ {rm.threshold:.2f}). Client assets in red are HIGH risk. Click the map, then ask the AI Analyst.")
    if out and out.get("last_clicked"):
        st.session_state["clicked"] = out["last_clicked"]

with tab_sat:
    st.subheader("Training label: observed inundation vs. the satellite proxy")
    pre, post = (Path(DATA_RAW / "satellite" / f"{s}_scene.txt").read_text().strip() for s in ("pre", "post"))
    st.caption(f"Sentinel-2 L2A pre-event {pre} and post-event {post}. Optical imagery cannot see through the storm "
               "clouds at the flood peak, so NDWI change only records what was still wet weeks later. The model is "
               "instead trained on the GSI inundation estimate published on 2020-07-04.")
    c1, c2, c3 = st.columns(3)
    raster_panel(c1, rm.layers["ndwi_pre"], "Pre-event NDWI", "BrBG", -0.8, 0.2, extent_km)
    raster_panel(c2, rm.layers["ndwi_change"], "NDWI change (post − pre)", "RdBu", -0.3, 0.3, extent_km)
    raster_panel(c3, np.where(rm.layers["flooded"] == 1, 1.0, np.nan), "GSI flood extent, 2020-07-04", "Blues", 0, 1.2, extent_km)
    px = metrics["proxy_label"]
    a, b, c = st.columns(3)
    a.metric("Proxy vs. GSI agreement (CSI)", f"{px['proxy_vs_reference']['csi']:.2f}")
    b.metric("Proxy recall of flooded cells", f"{px['proxy_vs_reference']['recall']:.0%}")
    c.metric("Model trained on proxy, PR-AUC vs. GSI", f"{px['rf_trained_on_proxy_scored_on_reference']['pr_auc']:.2f}",
             f"{px['rf_trained_on_proxy_scored_on_reference']['pr_auc'] - metrics['cv'][best]['spatial_cv']['pr_auc_mean']:+.2f} vs GSI-trained")

with tab_model:
    st.subheader("Cross-validated performance")
    rows = []
    for name, r in metrics["cv"].items():
        for scheme in ("spatial_cv", "random_cv"):
            rows.append({"model": name, "cv": scheme.replace("_cv", ""),
                         **{k: r[scheme][f"{k}_mean"] for k in ("pr_auc", "roc_auc", "f1", "csi", "precision", "recall")}})
    st.dataframe(pd.DataFrame(rows).set_index(["model", "cv"]).round(3))
    st.caption(f"Spatial CV holds out whole {metrics['data']['block_m'] // 1000} km blocks; random CV is shown only to "
               f"illustrate how much it overstates skill on autocorrelated rasters. Base rate "
               f"{metrics['data']['prevalence']:.1%} flooded. Selected: **{best}** (highest spatial PR-AUC).")
    st.warning(metrics["caveat"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Permutation importance (PR-AUC drop)**")
        imp = pd.Series(metrics["feature_importance"]).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        imp.plot.barh(ax=ax, color="#3b6ea5")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.markdown("**Reliability of calibrated out-of-fold scores**")
        rel = pd.DataFrame(metrics["calibration"]["reliability"])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
        ax.plot(rel.mean_predicted, rel.observed_rate, marker="o", color="#3b6ea5")
        ax.set_xlabel("predicted")
        ax.set_ylabel("observed flooded share")
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("Where the model is wrong (out-of-fold)")
    y = rm.layers["flooded"]
    pred = rm.risk >= rm.threshold
    err = np.full(y.shape, np.nan)
    err[np.isfinite(rm.risk)] = 0
    err[pred & (y == 1)] = 1
    err[pred & (y == 0) & np.isfinite(rm.risk)] = 2
    err[~pred & (y == 1) & np.isfinite(rm.risk)] = 3
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = ListedColormap(["#eeeeee", "#2ca02c", "#ff7f0e", "#d62728"])
    ax.imshow(err, cmap=cmap, vmin=-0.5, vmax=3.5, extent=extent_km, interpolation="nearest")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in
                       zip(cmap.colors, ["dry, correct", "flooded, caught", "false alarm", "missed"])], loc="lower left", fontsize=8)
    ax.set_xlabel("km east")
    st.pyplot(fig)
    plt.close(fig)

with tab_client:
    st.subheader("Client property portfolio (simulated values on real OSM building locations)")
    st.dataframe(portfolio.sort_values("risk_score", ascending=False), hide_index=True)
    st.download_button("Download scored portfolio CSV", portfolio.to_csv(index=False), "client_portfolio_scored.csv")
    st.subheader("Ingest a new client dataset")
    up = st.file_uploader("Upload CSV or GeoJSON", type=["csv", "geojson", "json"])
    if up:
        tmp = DATA_PROC / f"_upload{Path(up.name).suffix or '.csv'}"
        tmp.write_bytes(up.read())
        try:
            result = ingest_client_dataset(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        rep = result.report if hasattr(result, "report") else result[1]
        st.json(rep.to_dict())

with tab_agent:
    st.subheader("Natural-language flood-risk analyst")
    rain = max(get_event_rainfall()["points"].values(), key=lambda v: v["event_total_mm"])
    st.caption(f"Event rainfall {region['event_window']['start']} to {region['event_window']['end']}: "
               f"{rain['event_total_mm']:.0f} mm (Open-Meteo reanalysis). Known places: {', '.join(region['known_places'])}.")
    default_q = "Assess the flood risk around downtown Hitoyoshi after recent rainfall and identify exposed infrastructure within 2km."
    if "clicked" in st.session_state:
        c = st.session_state["clicked"]
        default_q = f"Assess the flood risk at {c['lat']:.4f}, {c['lng']:.4f} and identify exposed assets within 2km."
    q = st.text_area("Client request", default_q, height=80)
    if st.button("Run analysis", type="primary"):
        with st.spinner("Running agent pipeline..."):
            res = run_agent(q)
        st.caption(f"mode: {res['mode']} · latency: {res['latency_ms']} ms")
        st.markdown(res["report_markdown"])
        st.download_button("Download report", res["report_markdown"], "flood_risk_report.md")
        with st.expander("Agent tool-call log"):
            st.json(res["tool_log"])
