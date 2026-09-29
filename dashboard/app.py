import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

from src.config import DATA_RAW, DATA_PROC, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.agents.tools import get_region_info
from src.agents.agent import run_agent
from src.ml.model import RiskModel
from src.geo.terrain import load_grid
from src.geo.indices import ndwi
from src.ingestion.client_adapter import ingest_client_dataset

st.set_page_config(page_title="Inundata", layout="wide")

@st.cache_resource
def get_model():
    return RiskModel()

@st.cache_data
def get_grid_df():
    return get_model().predict_grid()

@st.cache_data
def get_ndwi_pair():
    lats, lons, _ = load_grid()
    return ndwi("pre_flood_2020-05-11", lats, lons), ndwi("post_flood_2020-08-29", lats, lons), lats, lons

def array_to_rgba(arr, cmap, vmin, vmax):
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap(cmap)(norm(np.nan_to_num(arr, nan=vmin)))
    rgba[np.isnan(arr)] = [0, 0, 0, 0]
    return rgba

st.title("Inundata — Kuma River Basin Flood Intelligence")
st.caption("Simulated client engagement · Hitoyoshi, Kumamoto, Japan · July 2020 flood disaster")

region = get_region_info()
df = get_grid_df()
client_df = pd.read_csv(DATA_RAW / "client" / "client_portfolio.csv")
infra = json.loads((DATA_RAW / "gis" / "infrastructure.geojson").read_text())
river = json.loads((DATA_RAW / "gis" / "kuma_river.geojson").read_text())
metrics = json.loads((DATA_PROC / "metrics.json").read_text())

tab_map, tab_sat, tab_model, tab_client, tab_agent = st.tabs(
    ["Risk Map", "Satellite Change Detection", "Model Performance", "Client Data", "AI Analyst"])

with tab_map:
    c1, c2 = st.columns([3, 1])
    with c2:
        st.metric("Assets in AOI", len(client_df))
        st.metric("Total exposure", f"¥{client_df.value_jpy.sum():,.0f}")
        st.metric("High-risk grid cells (score≥0.5)", int((df.risk_score >= 0.5).sum()))
        show_assets = st.checkbox("Show client assets", True)
        show_infra = st.checkbox("Show critical infrastructure", True)
        show_river = st.checkbox("Show river centerline", True)
    with c1:
        m = folium.Map(location=[(LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2], zoom_start=12, tiles="OpenStreetMap")
        rgba = array_to_rgba(df.pivot(index="lat", columns="lon", values="risk_score").sort_index(ascending=False).values, "RdYlBu_r", 0, 1)
        folium.raster_layers.ImageOverlay(rgba, bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]], opacity=0.55, name="Flood risk").add_to(m)
        if show_river:
            folium.GeoJson(river, name="Kuma River").add_to(m)
        if show_infra:
            for f in infra["features"]:
                lo, la = f["geometry"]["coordinates"]
                folium.CircleMarker([la, lo], radius=4, color="black", fill=True, fill_opacity=0.8,
                                     popup=f["properties"]["kind"]).add_to(m)
        if show_assets:
            for _, r in client_df.iterrows():
                folium.CircleMarker([r.lat, r.lon], radius=3, color="blue", fill=True, fill_opacity=0.6,
                                     popup=f"{r.asset_id} ¥{r.value_jpy:,}").add_to(m)
        folium.LayerControl().add_to(m)
        out = st_folium(m, height=560, width=None)
    st.caption("Red = higher modeled flood susceptibility. Click a point on the map, then ask the AI Analyst about that location.")
    if out and out.get("last_clicked"):
        st.session_state["clicked"] = out["last_clicked"]

with tab_sat:
    st.subheader("Sentinel-2 NDWI — before vs. after the July 2020 flood")
    st.caption("Pre-flood: 2020-05-11 (1.2% cloud) · Post-flood: 2020-08-29 (1.0% cloud). "
               "Optical imagery cannot see through the storm clouds present during the flood peak (2020-07-04); "
               "this compares standing-water/moisture signature before vs. seven weeks after the event — a documented limitation.")
    w1, w2, lats, lons = get_ndwi_pair()
    c1, c2, c3 = st.columns(3)
    for col, arr, title in [(c1, w1, "Pre-flood NDWI"), (c2, w2, "Post-flood NDWI"), (c3, w2 - w1, "Change (post − pre)")]:
        fig, ax = plt.subplots(figsize=(4, 3.3))
        im = ax.imshow(arr, cmap="BrBG" if "Change" not in title else "RdBu", vmin=-0.2 if "Change" in title else -0.8,
                        vmax=0.2 if "Change" in title else 0.1, extent=[lons.min(), lons.max(), lats.min(), lats.max()])
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04)
        col.pyplot(fig)
    st.info(f"Grid cells flagged as inundation-signal proxy (NDWI increase > 0.04): {int(df.inundation_signal.sum() if 'inundation_signal' in df else 0)} of {len(df)}")

with tab_model:
    st.subheader("Cross-validated performance (spatial GroupKFold, held-out blocks)")
    cv = metrics["cv_metrics"]
    st.dataframe(pd.DataFrame(cv).T)
    st.warning(metrics["caveat"])
    st.subheader("Feature importance")
    imp = pd.Series(metrics["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax, color="#3b6ea5")
    st.pyplot(fig)
    err_path = DATA_PROC / "error_analysis.csv"
    if err_path.exists():
        st.subheader("Error analysis (in-sample, for spatial inspection only)")
        st.dataframe(pd.read_csv(err_path))

with tab_client:
    st.subheader("Client property portfolio (simulated)")
    st.dataframe(client_df)
    st.download_button("Download portfolio CSV", client_df.to_csv(index=False), "client_portfolio.csv")
    st.subheader("Ingest a new client dataset")
    up = st.file_uploader("Upload CSV or GeoJSON", type=["csv", "geojson", "json"])
    if up:
        suffix = Path(up.name).suffix or ".csv"
        tmp = DATA_PROC / f"_upload{suffix}"
        tmp.write_bytes(up.read())
        result = ingest_client_dataset(tmp)
        rep = result.report if hasattr(result, "report") else result[1]
        st.json(rep.to_dict())

with tab_agent:
    st.subheader("Natural-language flood-risk analyst")
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
