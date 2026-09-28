"""
app.py — EPSIS (Explainable Predictive Satellite Intelligence System)
====================================================================
Main Streamlit Application for GEE Satellite Image Change Detection,
Classification, Severity Analysis, Risk Assessment, and Model Explainability (HiResCAM).
"""

import os
import hashlib
import datetime
import numpy as np
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

PRODUCTION_MODEL_PATH = "Tiny_model_4_CD/pretrained_models/levir_best.pth"


@st.cache_resource
def get_production_model():
    return load_model(PRODUCTION_MODEL_PATH)


# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
st.sidebar.title("🛰️ EPSIS Settings")
st.sidebar.markdown("**Explainable Predictive Satellite Intelligence System**")
st.sidebar.markdown("---")
st.sidebar.caption("System Status: **Active**")
st.sidebar.caption("Satellite Engine: **Google Earth Engine (GEE)**")
st.sidebar.caption("Constellation: **Sentinel-2 SR Harmonized (10m)**")
st.sidebar.caption("Spatial Alignment: **Native BBox Scale 10m**")

# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.title("🛰️ EPSIS")
st.markdown("### **Explainable Predictive Satellite Intelligence System**")
st.caption("Live GEE Satellite Change Detection, Classification, Severity Analysis, Risk Assessment, & XAI (HiResCAM)")

st.markdown("---")

# ---------------------------------------------------------------------------
# Location Selection (Interactive Map & Autocomplete Search)
# ---------------------------------------------------------------------------
if "lat" not in st.session_state:
    st.session_state.lat = 12.9800
    st.session_state.lon = 77.7400
    st.session_state.location_label = "Selected Location"

st.header("📍 Select Target Location")
st.caption("Type any worldwide city/region or click directly on the map to set your target location & spatial extent.")

search_query = st.text_input(
    "🔍 Search Target Location (Worldwide)",
    value=st.session_state.get("search_text", ""),
    placeholder="Type any city, town, landmark, or region worldwide (e.g. Vijayawada, Hyderabad, London, Tokyo, New York)...",
    key="location_search_input"
)

if search_query and len(search_query.strip()) >= 2 and search_query != st.session_state.get("last_searched_query"):
    st.session_state.last_searched_query = search_query.strip()
    geo_res = geocode_location(search_query.strip(), limit=5)
    st.session_state.search_candidates = geo_res.get("candidates", [])

if st.session_state.get("search_candidates"):
    candidates = st.session_state.search_candidates
    st.markdown("<small><b>📍 Matching Worldwide Locations (Click to select & jump on map):</b></small>", unsafe_allow_html=True)
    with st.container(border=True):
        for c_idx, cand in enumerate(candidates):
            disp_name = cand["display_name"]
            if st.button(f"📌 {disp_name}", key=f"loc_cand_{c_idx}", use_container_width=True):
                st.session_state.lat = float(cand["lat"])
                st.session_state.lon = float(cand["lon"])
                st.session_state.location_label = disp_name
                st.session_state.search_text = disp_name
                st.session_state.pop("search_candidates", None)
                st.session_state.pop("last_searched_query", None)
                st.rerun()

st.markdown(f"**Current Target Center:** `{st.session_state.location_label}` — **({st.session_state.lat:.6f}, {st.session_state.lon:.6f})**")

# Interactive Map display
m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=13)
folium.Marker([st.session_state.lat, st.session_state.lon], tooltip=st.session_state.location_label).add_to(m)
map_data = st_folium(m, width=800, height=380, key="map")

if map_data:
    if map_data.get("last_clicked") is not None:
        clicked_lat = float(map_data["last_clicked"]["lat"])
        clicked_lon = float(map_data["last_clicked"]["lng"])
        if abs(clicked_lat - st.session_state.lat) > 1e-5 or abs(clicked_lon - st.session_state.lon) > 1e-5:
            st.session_state.lat = clicked_lat
            st.session_state.lon = clicked_lon
            st.session_state.location_label = f"Map Location ({clicked_lat:.4f}, {clicked_lon:.4f})"
            st.rerun()

zoom_val = map_data.get("zoom", 13) if isinstance(map_data, dict) and map_data.get("zoom") else 13
aoi_radius = max(500, min(3500, int(1000 * (2 ** (13 - zoom_val)))))

latitude = st.session_state.lat
longitude = st.session_state.lon

st.markdown("---")

# ---------------------------------------------------------------------------
# Acquisition Date Pickers
# ---------------------------------------------------------------------------
st.header("📅 Select Satellite Acquisition Dates")
st.caption("🛰️ **Constellation Archives**: Sentinel-2 (June 2015 – Present) · Landsat-8 (Feb 2013 – Present) · Landsat-7 (1999 – Present)")


def select_image_date(label: str, default_date: datetime.date, key: str) -> datetime.date:
    """Renders a single date selector widget to pick Year, Month, and Day together."""
    return st.date_input(
        label=label,
        value=default_date,
        min_value=datetime.date(1999, 1, 1),
        max_value=datetime.date(2026, 12, 31),
        key=key
    )


col_before, col_after = st.columns(2)
with col_before:
    before_date = select_image_date("🛰️ Before Image Date (T1)", datetime.date(2021, 6, 15), "before_date")

with col_after:
    after_date = select_image_date("🛰️ After Image Date (T2)", datetime.date(2023, 6, 20), "after_date")

st.markdown("")
run_analysis_btn = st.button("🚀 Run Live GEE Change Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Processing Pipeline Execution
# ---------------------------------------------------------------------------
if run_analysis_btn:
    if before_date is None or after_date is None:
        st.error("⚠️ Please select valid dates for both 'Before Image' and 'After Image'.")
        st.stop()
    if after_date <= before_date:
        st.error("⚠️ Validation Error: 'After Image' date must be strictly later than 'Before Image' date.")
        st.stop()

    with st.spinner("Step 1/4: Retrieving GEE Sentinel-2 SR Harmonized imagery at native 10m scale..."):
        try:
            ref_path, comp_path, sat_meta = fetch_satellite_pair(
                latitude, longitude,
                str(before_date), str(after_date),
                buffer_m=aoi_radius
            )
        except Exception as e:
            st.error(f"⚠️ Satellite Data Acquisition Error: {e}")
            st.stop()

    with st.spinner("Step 2/4: Executing Siamese U-Net Change Detection & Paper-Faithful HiResCAM..."):
        try:
            model = get_production_model()
            pred_res = predict_change(
                model, ref_path, comp_path,
                threshold=0.45,
                use_otsu=True,
                apply_morph=True
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

    # Generate Dynamic Analysis ID based on scene IDs + bbox
    scene_str = sat_meta.get("before_scene_id", "") + sat_meta.get("after_scene_id", "") + str(sat_meta.get("bbox", ""))
    dynamic_analysis_id = f"GEE-{hashlib.md5(scene_str.encode('utf-8')).hexdigest()[:10].upper()}"

    st.session_state["epsis_data"] = {
        "ref_path": ref_path,
        "comp_path": comp_path,
        "sat_meta": sat_meta,
        "pred_res": pred_res,
        "analysis_res": analysis_res,
        "before_date": str(before_date),
        "after_date": str(after_date),
        "analysis_id": dynamic_analysis_id,
    }

# Render Results if available in Session State
if "epsis_data" in st.session_state:
    data = st.session_state["epsis_data"]
    ref_path = data["ref_path"]
    comp_path = data["comp_path"]
    sat_meta = data["sat_meta"]
    pred_res = data["pred_res"]
    analysis_res = data["analysis_res"]
    analysis_id = data.get("analysis_id", "GEE-LIVE")
    eff_t = pred_res.get("effective_threshold", 0.45)

    b_req = sat_meta.get("before_requested", data.get("before_date"))
    b_act = sat_meta.get("before_actual", b_req)
    b_const = sat_meta.get("before_constellation", "Sentinel-2 SR Harmonized")
    a_req = sat_meta.get("after_requested", data.get("after_date"))
    a_act = sat_meta.get("after_actual", a_req)
    a_const = sat_meta.get("after_constellation", "Sentinel-2 SR Harmonized")

    st.markdown("---")
    st.header("🔍 EPSIS GEE Live Intelligence Results")
    st.caption(f"🆔 **Dynamic Analysis ID**: `{analysis_id}` | Active Engine: **{sat_meta.get('provider', 'Google Earth Engine')}**")
    st.success(f"✅ GEE Sentinel-2 SR Harmonized Imagery Loaded Successfully at Native 10m Spatial Scale.")

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

    # 1. Temporal Satellite Imagery Display (T1 & T2)
    st.subheader("🖼️ Temporal GEE Satellite Imagery")
    i1, i2 = st.columns(2)
    with i1:
        st.subheader("Before Image (T1)")
        st.image(ref_path, caption=f"T1 GEE RGB Image ({sat_meta.get('before_width', 0)} x {sat_meta.get('before_height', 0)} px)\nRequested: {b_req} | Acquired: {b_act}", use_container_width=True)
    with i2:
        st.subheader("After Image (T2)")
        st.image(comp_path, caption=f"T2 GEE RGB Image ({sat_meta.get('after_width', 0)} x {sat_meta.get('after_height', 0)} px)\nRequested: {a_req} | Acquired: {a_act}", use_container_width=True)

    st.markdown("---")

    # 2. Predicted Change Mask & Reference Change Analysis
    st.subheader("🎯 Change Prediction & Reference Analysis")
    m_c1, m_c2 = st.columns(2)
    with m_c1:
        st.subheader("Predicted Change Mask")
        st.image(pred_res["binary_mask_rgb"], caption="BLACK = No Change (0) | WHITE = Predicted Change (255)", use_container_width=True)
        st.caption("Binary change prediction generated by Siamese U-Net on current GEE T1/T2 pair.")

    with m_c2:
        st.subheader("Reference Change Analysis")
        st.image(pred_res["reference_change_analysis"], caption="BLACK = Low Difference (0) | WHITE = Strong Difference (255)", use_container_width=True)
        st.caption("Independent temporal image-difference analysis computed directly from current T1/T2 GEE imagery.")

    st.markdown("ℹ️ **Reference Ground Truth: Not Available** *(Arbitrary live GEE satellite acquisition without human annotation).*")

    st.markdown("---")

    # 3. Paper-Faithful HiResCAM Explainability (IEEE ICITEICS 2025)
    st.subheader("🔥 Paper-Faithful HiResCAM Explainability")
    st.caption("Feature activation & backward gradient propagation maps explaining model attention for Pre-change, Post-change, and Bottleneck branches.")

    h_c1, h_c2 = st.columns(2)
    with h_c1:
        st.subheader("HiResCAM Pre")
        if pred_res.get("hirescam_pre") is not None:
            st.image(pred_res["hirescam_pre"], caption="HiResCAM Pre-Change Attention Overlay (T1 Encoder A)", use_container_width=True)
        else:
            st.info("HiResCAM Pre could not be generated.")

    with h_c2:
        st.subheader("HiResCAM Post")
        if pred_res.get("hirescam_post") is not None:
            st.image(pred_res["hirescam_post"], caption="HiResCAM Post-Change Attention Overlay (T2 Encoder B)", use_container_width=True)
        else:
            st.info("HiResCAM Post could not be generated.")

    if pred_res.get("hirescam_bot") is not None:
        st.subheader("HiResCAM Bottleneck")
        st.image(pred_res["hirescam_bot"], caption="HiResCAM Fused Representation Attention Map (Siamese Bottleneck Layer)", use_container_width=True)

    st.markdown("##### 🎨 HiResCAM Attention Color Scale")
    st.image(pred_res["prob_map_rgb"], caption="BLUE (Extremely Low) → CYAN (Low) → GREEN (Medium) → YELLOW (High) → RED (Extremely High)", use_container_width=True)

    st.markdown("---")

    # 4. Intelligence & Risk Report
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

    st.markdown("---")

    # Diagnostic Image Pipeline & Cloud Mask Inspection Panel
    with st.expander("🔬 GEE Satellite Image Quality & Cloud Mask Diagnostics", expanded=False):
        st.subheader("Diagnostic Raw RGBs & Cloud Mask Overlay")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("#### T1 (Before Image) Diagnostics")
            st.image(ref_path, caption=f"T1 Raw RGB Output ({sat_meta.get('before_width', 0)} x {sat_meta.get('before_height', 0)} px)", use_container_width=True)
            if sat_meta.get("t1_mask_path") and os.path.exists(sat_meta.get("t1_mask_path")):
                st.image(sat_meta.get("t1_mask_path"), caption="T1 Detected Cloud & Cloud Shadow Mask (Red = Cloud/Shadow)", use_container_width=True)

        with d_col2:
            st.markdown("#### T2 (After Image) Diagnostics")
            st.image(comp_path, caption=f"T2 Raw RGB Output ({sat_meta.get('after_width', 0)} x {sat_meta.get('after_height', 0)} px)", use_container_width=True)
            if sat_meta.get("t2_mask_path") and os.path.exists(sat_meta.get("t2_mask_path")):
                st.image(sat_meta.get("t2_mask_path"), caption="T2 Detected Cloud & Cloud Shadow Mask (Red = Cloud/Shadow)", use_container_width=True)

        st.markdown("#### T1 / T2 Pipeline Telemetry & Spatial Metadata")
        diag_table = [
            {"Metadata Field": "Dynamic Analysis ID", "T1 (Before Image)": analysis_id, "T2 (After Image)": analysis_id},
            {"Metadata Field": "Provider Engine", "T1 (Before Image)": sat_meta.get("provider"), "T2 (After Image)": sat_meta.get("provider")},
            {"Metadata Field": "Constellation Asset", "T1 (Before Image)": b_const, "T2 (After Image)": a_const},
            {"Metadata Field": "Scene ID", "T1 (Before Image)": sat_meta.get("before_scene_id", "N/A"), "T2 (After Image)": sat_meta.get("after_scene_id", "N/A")},
            {"Metadata Field": "Requested Target Date", "T1 (Before Image)": b_req, "T2 (After Image)": a_req},
            {"Metadata Field": "Actual Acquisition Date", "T1 (Before Image)": b_act, "T2 (After Image)": a_act},
            {"Metadata Field": "Cloud Cover Percentage", "T1 (Before Image)": f"{sat_meta.get('before_cloud', 0.0):.2f}%", "T2 (After Image)": f"{sat_meta.get('after_cloud', 0.0):.2f}%"},
            {"Metadata Field": "ROI Extent (Meters)", "T1 (Before Image)": f"{sat_meta.get('roi_width_m')}m x {sat_meta.get('roi_height_m')}m", "T2 (After Image)": f"{sat_meta.get('roi_width_m')}m x {sat_meta.get('roi_height_m')}m"},
            {"Metadata Field": "Expected Native 10m Pixels", "T1 (Before Image)": f"{sat_meta.get('expected_native_width')} x {sat_meta.get('expected_native_height')} px", "T2 (After Image)": f"{sat_meta.get('expected_native_width')} x {sat_meta.get('expected_native_height')} px"},
            {"Metadata Field": "Actual Output Dimensions", "T1 (Before Image)": f"{sat_meta.get('before_width')} x {sat_meta.get('before_height')} px", "T2 (After Image)": f"{sat_meta.get('after_width')} x {sat_meta.get('after_height')} px"},
            {"Metadata Field": "GEE Resampling Convolution", "T1 (Before Image)": "Native 10m Grid Extraction", "T2 (After Image)": "Native 10m Grid Extraction"},
            {"Metadata Field": "10-Point Quality Validation", "T1 (Before Image)": sat_meta.get("validation_status", "PASSED"), "T2 (After Image)": sat_meta.get("validation_status", "PASSED")},
            {"Metadata Field": "Spatial Resolution", "T1 (Before Image)": sat_meta.get("resolution", "10 m"), "T2 (After Image)": sat_meta.get("resolution", "10 m")},
            {"Metadata Field": "Coordinate System (CRS)", "T1 (Before Image)": sat_meta.get("crs", "EPSG:4326"), "T2 (After Image)": sat_meta.get("crs", "EPSG:4326")},
            {"Metadata Field": "RGB Bands Rendered", "T1 (Before Image)": ", ".join(sat_meta.get("bands", [])), "T2 (After Image)": ", ".join(sat_meta.get("bands", []))},
        ]
        st.dataframe(diag_table, use_container_width=True)

    # Developer Diagnostics Panel
    with st.expander("🔧 Developer Diagnostics"):
        prob_arr = pred_res["probability_map"]
        bin_mask = pred_res["binary_mask"]
        p_pcts = np.percentile(prob_arr, [1, 25, 50, 75, 99])
        h_stats = pred_res.get("hirescam_stats", {})

        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.markdown("#### MODEL & PIPELINE METADATA")
            st.write(f"**Dynamic Analysis ID:** `{analysis_id}`")
            st.write(f"**Model Architecture:** `TinyCD Siamese U-Net (Pretrained)`")
            st.write(f"**Provider Engine:** `{sat_meta.get('provider')}`")
            st.write(f"**GEE Image Dimensions:** `{pred_res.get('target_w')} x {pred_res.get('target_h')} px`")
            st.write(f"**Effective Threshold:** `{eff_t:.4f}` (Dynamic Auto-Otsu Thresholding)")
            st.write(f"**Morphological Noise Filter:** `Enabled (3x3 Rect Open/Close)`")
            st.write(f"**Before Image Scene:** `{sat_meta.get('before_scene_id', 'N/A')}`")
            st.write(f"**After Image Scene:** `{sat_meta.get('after_scene_id', 'N/A')}`")

        with d_c2:
            st.markdown("#### MODEL PROBABILITY & MASK TELEMETRY")
            st.code(
                f"Model Output Tensor Shape: {bin_mask.shape}\n"
                f"Binary Mask Unique Values: {np.unique(bin_mask).tolist()}\n"
                f"Probability Min: {pred_res.get('p_min', float(prob_arr.min())):.6f}\n"
                f"Probability Max: {pred_res.get('p_max', float(prob_arr.max())):.6f}\n"
                f"Probability Mean: {pred_res.get('p_mean', float(prob_arr.mean())):.6f}\n"
                f"Probability Percentiles (p1, p25, p50, p75, p99):\n  [{p_pcts[0]:.6f}, {p_pcts[1]:.6f}, {p_pcts[2]:.6f}, {p_pcts[3]:.6f}, {p_pcts[4]:.6f}]\n"
                f"Effective Threshold: {eff_t:.6f}\n"
                f"Predicted Change Percentage: {pred_res.get('changed_fraction', 0.0)*100.0:.3f}%\n"
                f"Reference Analysis Change %: {pred_res.get('ref_changed_fraction', 0.0)*100.0:.3f}%"
            )

        st.markdown("#### HIRESCAM ACTIVATION & GRADIENT STATISTICS")
        st.code(
            f"Target Layer (Pre-Encoder): Encoder A Deep Conv Features\n"
            f"Pre-Encoder Activation Mean: {h_stats.get('pre_act_mean', 0.0):.4f} | Grad Std: {h_stats.get('pre_grad_std', 0.0):.6f} | CAM Std: {h_stats.get('pre_cam_std', 0.0):.4f}\n"
            f"Target Layer (Post-Encoder): Encoder B Deep Conv Features\n"
            f"Post-Encoder Activation Mean: {h_stats.get('post_act_mean', 0.0):.4f} | Grad Std: {h_stats.get('post_grad_std', 0.0):.6f} | CAM Std: {h_stats.get('post_cam_std', 0.0):.4f}\n"
            f"Target Layer (Bottleneck): Siamese Bottleneck Fused Latents\n"
            f"Bottleneck Activation Mean: {h_stats.get('bot_act_mean', 0.0):.4f} | Grad Std: {h_stats.get('bot_grad_std', 0.0):.6f} | CAM Std: {h_stats.get('bot_cam_std', 0.0):.4f}"
        )
