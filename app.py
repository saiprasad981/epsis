"""
app.py — EPSIS (Explainable Predictive Satellite Intelligence System)
====================================================================
Main Streamlit Application for Satellite Image Change Detection,
Classification, Quantitative Severity Analysis, Risk Assessment, and
Model Explainability (HiResCAM).
"""

import datetime
import streamlit as st
import folium
from streamlit_folium import st_folium

import importlib
import satellite
import inference
import geocode
import epsis_analysis

importlib.reload(satellite)
importlib.reload(inference)
importlib.reload(geocode)
importlib.reload(epsis_analysis)

from satellite import fetch_satellite_pair
from inference import load_model, predict_change
from geocode import geocode_location
from epsis_analysis import analyze_detected_changes

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EPSIS — Satellite Change Detection & Risk Assessment",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

LEVIR_MODEL_PATH = "Tiny_model_4_CD/pretrained_models/levir_best.pth"
WHU_MODEL_PATH = "Tiny_model_4_CD/pretrained_models/whu_best.pth"


@st.cache_resource
def get_cached_model(model_choice: str):
    path = LEVIR_MODEL_PATH if "LEVIR" in model_choice else WHU_MODEL_PATH
    return load_model(path)


# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
st.sidebar.title("🛰️ EPSIS Controls")
st.sidebar.markdown("**Explainable Predictive Satellite Intelligence System**")

model_choice = st.sidebar.selectbox(
    "TinyCD Checkpoint",
    ["LEVIR-CD (Best for Urban / Building Change)", "WHU-CD (Best for High-Res Structures)"],
    index=0
)

use_otsu = st.sidebar.checkbox("Auto-Calculate Optimal Otsu Threshold", value=True, help="Automatically calculates statistically optimal bimodal threshold for scene activations.")
threshold = st.sidebar.slider(
    "Manual Change Threshold Fallback",
    min_value=0.01, max_value=0.90, value=0.45, step=0.01,
    help="Used when Auto-Otsu is disabled or as lower bound fallback."
)

apply_morph = st.sidebar.checkbox("Apply Morphological Noise Filter", value=True)
aoi_radius = st.sidebar.slider("Area of Interest Radius (meters)", min_value=500, max_value=3000, value=1000, step=250)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Location Presets")
preset = st.sidebar.selectbox(
    "Quick Test Locations",
    ["Bengaluru Tech Park (India)", "Kokapet Growth Corridor (Hyderabad)", "Hyderabad Hitech City (India)", "LEVIR Building Development", "Custom / Search"],
    index=0
)

if preset == "Bengaluru Tech Park (India)":
    st.session_state.lat, st.session_state.lon = 12.9800, 77.7400
    st.session_state.location_label = "Bengaluru Whitefield"
elif preset == "Kokapet Growth Corridor (Hyderabad)":
    st.session_state.lat, st.session_state.lon = 17.3950, 78.3300
    st.session_state.location_label = "Kokapet Financial District"
elif preset == "Hyderabad Hitech City (India)":
    st.session_state.lat, st.session_state.lon = 17.4475, 78.3762
    st.session_state.location_label = "Hyderabad Hitech City"
elif preset == "LEVIR Building Development":
    st.session_state.lat, st.session_state.lon = 32.2226, -110.9747
    st.session_state.location_label = "Urban Development Zone"

# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.title("🛰️ EPSIS")
st.markdown("### **Explainable Predictive Satellite Intelligence System**")
st.caption("Integrated Change Detection, Classification, Severity Analysis, Risk Assessment, and HiResCAM Model Explainability")

st.markdown("---")

# ---------------------------------------------------------------------------
# Location Selection
# ---------------------------------------------------------------------------
if "lat" not in st.session_state:
    st.session_state.lat = 12.9800
    st.session_state.lon = 77.7400
    st.session_state.location_label = "Bengaluru Whitefield (Default)"

st.header("📍 Select Target Location")

search_col, button_col = st.columns([4, 1])
with search_col:
    search_query = st.text_input(
        "Search Location",
        placeholder="e.g. Hitech City Hyderabad · Bengaluru · Kokapet · London",
        label_visibility="collapsed",
    )
with button_col:
    search_clicked = st.button("🔍 Search Location", use_container_width=True)

if search_clicked and search_query:
    geo_res = geocode_location(search_query, limit=5)
    candidates = geo_res.get("candidates", [])
    if candidates:
        st.session_state.search_candidates = candidates
    else:
        st.warning(f"No exact match found for '{search_query}'. Try a city, district, or landmark name.")

if st.session_state.get("search_candidates"):
    candidates = st.session_state.search_candidates
    options = [f"{c['display_name']} ({c['place_type']})" for c in candidates]
    chosen_idx = st.selectbox("Select match:", range(len(options)), format_func=lambda i: options[i])
    if st.button("✅ Confirm Location Selection"):
        chosen = candidates[chosen_idx]
        st.session_state.lat = chosen["lat"]
        st.session_state.lon = chosen["lon"]
        st.session_state.location_label = chosen["display_name"]
        st.session_state.pop("search_candidates", None)
        st.rerun()

st.caption(f"Target Center: **{st.session_state.location_label}** ({st.session_state.lat:.6f}, {st.session_state.lon:.6f})")

# Map display
m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=13)
folium.Marker([st.session_state.lat, st.session_state.lon], tooltip=st.session_state.location_label).add_to(m)
map_data = st_folium(m, width=800, height=350, key="map")

if map_data.get("last_clicked") is not None:
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]
    if clicked_lat != st.session_state.lat or clicked_lon != st.session_state.lon:
        st.session_state.lat = clicked_lat
        st.session_state.lon = clicked_lon
        st.session_state.location_label = "Custom Pin Coordinates"
        st.rerun()

latitude = st.session_state.lat
longitude = st.session_state.lon

st.markdown("---")

# ---------------------------------------------------------------------------
# Temporal Range Pickers
# ---------------------------------------------------------------------------
st.header("📅 Select Temporal Comparison Window")

col_ref, col_comp = st.columns(2)
with col_ref:
    st.subheader("🛰️ Reference Period (T1)")
    r_c1, r_c2 = st.columns(2)
    with r_c1:
        ref_start = st.date_input("T1 Start", value=datetime.date(2021, 1, 1))
    with r_c2:
        ref_end = st.date_input("T1 End", value=datetime.date(2021, 12, 31))

with col_comp:
    st.subheader("🛰️ Comparison Period (T2)")
    c_c1, c_c2 = st.columns(2)
    with c_c1:
        comp_start = st.date_input("T2 Start", value=datetime.date(2023, 1, 1))
    with c_c2:
        comp_end = st.date_input("T2 End", value=datetime.date(2023, 12, 31))

st.markdown("")
run_analysis_btn = st.button("🚀 Run Complete EPSIS Intelligence Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Processing Pipeline Execution
# ---------------------------------------------------------------------------
if run_analysis_btn:
    with st.spinner("Step 1/4: Retrieving & spatially registering satellite imagery..."):
        try:
            ref_path, comp_path, sat_meta = fetch_satellite_pair(
                latitude, longitude,
                str(ref_start), str(ref_end),
                str(comp_start), str(comp_end),
                buffer_m=aoi_radius
            )
        except Exception as e:
            st.error(f"⚠️ Satellite Data Acquisition Error: {e}")
            st.stop()

    with st.spinner("Step 2/4: Executing TinyCD Deep Learning Change Detection & HiResCAM..."):
        try:
            model = get_cached_model(model_choice)
            pred_res = predict_change(
                model, ref_path, comp_path,
                threshold=threshold,
                use_otsu=use_otsu,
                apply_morph=apply_morph
            )
        except Exception as e:
            st.error(f"⚠️ Model Inference Error: {e}")
            st.stop()

    with st.spinner("Step 3/4: Calculating Classification, Severity, & Risk Metrics..."):
        try:
            analysis_res = analyze_detected_changes(
                ref_path, comp_path,
                pred_res["probability_map"],
                pred_res["binary_mask"],
                latitude, longitude
            )
        except Exception as e:
            st.error(f"⚠️ Analysis Computation Error: {e}")
            st.stop()

    # Save to session state for persistent rendering across tabs
    st.session_state["epsis_data"] = {
        "ref_path": ref_path,
        "comp_path": comp_path,
        "sat_meta": sat_meta,
        "pred_res": pred_res,
        "analysis_res": analysis_res,
        "ref_dates": (str(ref_start), str(ref_end)),
        "comp_dates": (str(comp_start), str(comp_end)),
    }

# Render Results if available in Session State
if "epsis_data" in st.session_state:
    data = st.session_state["epsis_data"]
    ref_path = data["ref_path"]
    comp_path = data["comp_path"]
    sat_meta = data["sat_meta"]
    pred_res = data["pred_res"]
    analysis_res = data["analysis_res"]
    r_start, r_end = data["ref_dates"]
    c_start, c_end = data["comp_dates"]

    st.markdown("---")
    st.header("🔍 EPSIS Intelligence Results")
    st.success(f"✅ Satellite Imagery Active via **{sat_meta.get('provider', 'Satellite Engine')}**")

    sev = analysis_res["severity"]
    risk = analysis_res["risk"]
    cls_res = analysis_res["classification"]

    # Top Summary Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Changed Area (%)", f"{analysis_res['changed_percent']}%")
    m2.metric("Changed Extent", f"{analysis_res['changed_area_ha']} ha ({analysis_res['changed_area_m2']} m²)")
    m3.metric("Severity Score", f"{sev['score']} / 100", delta=sev['level'])
    m4.metric("Primary Class", cls_res["primary_class"])

    st.markdown("---")

    # Satellite Imagery & Detection Overlay Display
    st.subheader("🖼️ Temporal Satellite Imagery & Detection Overlay")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.subheader("Reference Image (T1)")
        st.image(ref_path, caption=f"T1 ({r_start} to {r_end})", use_container_width=True)
    with i2:
        st.subheader("Comparison Image (T2)")
        st.image(comp_path, caption=f"T2 ({c_start} to {c_end})", use_container_width=True)
    with i3:
        st.subheader("Change Highlight Overlay")
        st.image(pred_res["overlay_rgb"], caption="Red = Detected Temporal Change", use_container_width=True)

    st.markdown("---")

    # Visual Multi-Map Tabs
    st.subheader("📊 Visual Maps & Explainability (XAI)")
    t1, t2, t3, t4 = st.tabs(["Binary Change Mask", "JET Probability Map", "HiResCAM Model Explainability", "Model Confidence Map"])

    eff_t = pred_res.get("effective_threshold", threshold)

    with t1:
        st.image(pred_res["binary_mask_rgb"], caption=f"Binary Change Mask (Effective Threshold={eff_t:.4f}) — White = Changed Pixels", use_container_width=True)
    with t2:
        st.image(pred_res["prob_map_rgb"], caption="Raw TinyCD Probability Activation — Blue = Low Probability, Red = High Probability", use_container_width=True)
    with t3:
        st.image(pred_res["hirescam_rgb"], caption="HiResCAM Neural Network Feature Map Importance — Hotter colors indicate features driving model decision", use_container_width=True)
    with t4:
        st.image(pred_res["confidence_rgb"], caption="Model Confidence Map — Bright = Sure, Dark = Decision Boundary", use_container_width=True)

    # Diagnostic Statistics Panel
    with st.expander("📊 Map Diagnostic Metrics & Statistics"):
        prob_arr = pred_res["probability_map"]
        bin_mask = pred_res["binary_mask"]

        import numpy as np
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.markdown("#### ------------------------------\nBINARY MASK\n------------------------------")
            st.code(f"Shape: {bin_mask.shape}\nDtype: {bin_mask.dtype}\nUnique values: {np.unique(bin_mask).tolist()}\nChanged pixels: {int(bin_mask.sum())}\nValid pixels: {bin_mask.size}\nChanged %: {analysis_res['changed_percent']}%")

            st.markdown("#### ------------------------------\nPROBABILITY MAP\n------------------------------")
            st.code(f"Shape: {prob_arr.shape}\nMin: {prob_arr.min():.6f}\nMax: {prob_arr.max():.6f}\nMean: {prob_arr.mean():.6f}\nStd: {prob_arr.std():.6f}\nPercentiles (p1, p25, p50, p75, p99):\n{np.percentile(prob_arr, [1, 25, 50, 75, 99])}")

        with d_c2:
            st.markdown("#### ------------------------------\nHIRESCAM\n------------------------------")
            cam_rgb = pred_res["hirescam_rgb"]
            st.code(f"Shape: {cam_rgb.shape}\nDtype: {cam_rgb.dtype}\nMin RGB: {cam_rgb.min()}\nMax RGB: {cam_rgb.max()}\nMean RGB: {cam_rgb.mean():.2f}\nStd RGB: {cam_rgb.std():.2f}")

            st.markdown("#### ------------------------------\nCONFIDENCE MAP\n------------------------------")
            denom = max(eff_t, 1.0 - eff_t, 1e-5)
            conf_arr = np.abs(prob_arr - eff_t) / denom
            st.code(f"Shape: {conf_arr.shape}\nMin: {conf_arr.min():.6f}\nMax: {conf_arr.max():.6f}\nMean: {conf_arr.mean():.6f}\nStd: {conf_arr.std():.6f}")

    st.markdown("---")

    # Intelligence & Risk Report
    st.header("📋 EPSIS Predictive Intelligence Report")

    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.subheader("🎯 Change Classification")
        st.markdown(f"**Primary Category:** `{cls_res['primary_class']}`")
        st.write(cls_res["description"])
        st.markdown("**Class Probability Breakdown:**")
        for k, v in cls_res["breakdown"].items():
            st.progress(float(v), text=f"{k}: {v*100:.1f}%")

    with r_col2:
        st.subheader("⚠️ Severity & Risk Assessment")
        sev_color = sev["color"]
        sev_level_upper = sev["level"].upper()
        sev_score = sev["score"]
        st.markdown(f"### <span style='color:{sev_color};'>{sev_level_upper} SEVERITY ({sev_score}/100)</span>", unsafe_allow_html=True)
        st.write(sev["description"])

        risk_color = risk["risk_color"]
        risk_title = risk["risk_title"]
        st.markdown(f"#### Risk Status: <span style='color:{risk_color};'>{risk_title}</span>", unsafe_allow_html=True)
        st.write(risk["overview"])

        st.markdown("**Recommended Mitigation & Intervention Actions:**")
        for act in risk["action_items"]:
            st.markdown(f"- 📌 {act}")

    # Region Breakdown Table
    if analysis_res["num_regions"] > 0:
        st.markdown("---")
        st.subheader("🧩 Detected Region Breakdown")
        st.caption(f"Identified {analysis_res['num_regions']} distinct change cluster(s)")

        region_table = []
        for r in analysis_res["regions"][:10]:
            region_table.append({
                "Region ID": f"Region #{r['region_id']}",
                "Pixel Count": r["pixel_area"],
                "Area (Hectares)": round(r["hectares"], 3),
                "Area (m²)": round(r["area_m2"], 1),
                "Centroid (X, Y)": f"({r['centroid'][0]}, {r['centroid'][1]})",
                "Mean Confidence": f"{r['mean_prob']*100:.1f}%"
            })
        st.dataframe(region_table, use_container_width=True)

    with st.expander("🐛 Raw Pipeline Execution Logs"):
        st.write(f"Provider: `{sat_meta.get('provider')}`")
        st.write(f"Target BBox: `{sat_meta.get('bbox')}`")
        st.write(f"Raw Probability Range: [{pred_res['probability_map'].min():.4f}, {pred_res['probability_map'].max():.4f}]")
        st.write(f"Changed Pixels: {analysis_res['changed_pixels']} / {analysis_res['total_pixels']}")
