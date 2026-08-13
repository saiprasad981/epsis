import streamlit as st
import folium
from streamlit_folium import st_folium

from gee_utils import get_satellite_image, get_image_thumbnail, save_temp_image
from inference import load_model, predict_change
from geocode import geocode_location

st.set_page_config(
    page_title="EPSIS",
    page_icon="🛰️",
    layout="wide"
)

MODEL_PATH = "Tiny_model_4_CD/pretrained_models/levir_best.pth"


# ---------------------------------------------------------------------------
# Model loading — cached so it only loads ONCE per session, not on every
# widget interaction. Streamlit reruns the whole script top-to-bottom on
# every click/input change; without this cache, TinyCD (EfficientNet-B4
# backbone) would reload from disk every single time, which is slow and
# unnecessary since the model doesn't change between runs.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_cached_model():
    return load_model(MODEL_PATH)


model = get_cached_model()

st.title("🛰️ EPSIS")
st.subheader("Explainable Predictive Satellite Intelligence System")
st.caption("Change detection powered by TinyCD (pretrained on LEVIR-CD)")

st.markdown("---")

st.header("📍 Select Location")

# Session state holds the "current" location so it survives Streamlit's
# rerun-on-every-interaction behavior. Defaults to Hyderabad.
if "lat" not in st.session_state:
    st.session_state.lat = 17.3850
    st.session_state.lon = 78.4867
    st.session_state.location_label = "Hyderabad (default)"

search_col, button_col = st.columns([4, 1])
with search_col:
    search_query = st.text_input(
        "Search for a place",
        placeholder="e.g. Hitech City, Hyderabad  ·  Bengaluru  ·  Charminar",
        label_visibility="collapsed",
    )
with button_col:
    search_clicked = st.button("🔍 Search", use_container_width=True)

if search_clicked and search_query:
    try:
        geo_result = geocode_location(search_query, limit=5)
        candidates = geo_result["candidates"]
        errors = geo_result["errors"]
        reachable = geo_result["any_backend_reachable"]

        if candidates:
            st.session_state.search_candidates = candidates
        elif not reachable:
            st.error(
                "⚠️ Both geocoding services failed to respond — this looks like a real "
                "network/connectivity issue, not a 'place doesn't exist' issue:\n\n" +
                "\n".join(f"- {e}" for e in errors)
            )
            st.session_state.pop("search_candidates", None)
        else:
            st.warning(
                f"No match found for '{search_query}'. This service is reachable and "
                f"responded successfully — it just doesn't have this exact place mapped. "
                f"Try a shorter/simpler query (e.g. just the village or mandal name, "
                f"without institute names or extra qualifiers)."
            )
            st.session_state.pop("search_candidates", None)
    except Exception as e:
        st.error(f"⚠️ Unexpected error during geocoding: {e}")

if st.session_state.get("search_candidates"):
    candidates = st.session_state.search_candidates
    options = [f"{c['display_name']}  ·  ({c['place_type']})" for c in candidates]
    chosen_idx = st.selectbox(
        f"Found {len(candidates)} match(es) — select the correct one:",
        range(len(options)),
        format_func=lambda i: options[i],
    )
    chosen = candidates[chosen_idx]

    if st.button("✅ Use this location"):
        st.session_state.lat = chosen["lat"]
        st.session_state.lon = chosen["lon"]
        st.session_state.location_label = chosen["display_name"]
        st.session_state.pop("search_candidates", None)
        st.rerun()

st.caption(f"Current location: **{st.session_state.location_label}**  "
           f"({st.session_state.lat:.6f}, {st.session_state.lon:.6f})")

m = folium.Map(
    location=[st.session_state.lat, st.session_state.lon],
    zoom_start=12,
    control_scale=True
)
folium.Marker(
    [st.session_state.lat, st.session_state.lon],
    tooltip=st.session_state.location_label,
).add_to(m)

map_data = st_folium(
    m,
    width=700,
    height=500,
    key="map"
)

# Clicking the map overrides the search result — last interaction wins
if map_data.get("last_clicked") is not None:
    st.write(map_data["last_clicked"])
    st.session_state.lat = map_data["last_clicked"]["lat"]
    st.session_state.lon = map_data["last_clicked"]["lng"]
    st.session_state.location_label = "Custom point (map click)"

    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]

    if (
        clicked_lat != st.session_state.lat
        or clicked_lon != st.session_state.lon
    ):
        st.session_state.lat = clicked_lat
        st.session_state.lon = clicked_lon
        st.session_state.location_label = "Custom point (map click)"
        st.rerun()
latitude = st.session_state.lat
longitude = st.session_state.lon

st.markdown("## 🛰 Reference Period")

col1, col2 = st.columns(2)
with col1:
    ref_start = st.date_input("Reference Start Date")
with col2:
    ref_end = st.date_input("Reference End Date")

st.markdown("---")

st.markdown("## 🛰 Comparison Period")

col3, col4 = st.columns(2)
with col3:
    comp_start = st.date_input("Comparison Start Date")
with col4:
    comp_end = st.date_input("Comparison End Date")

st.markdown("")

threshold = st.slider(
    "Change detection threshold",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="TinyCD's raw output is a probability per pixel. Pixels above this "
         "value are classified as 'changed'. 0.5 is the model's own default."
)

fetch = st.button("🛰️ Fetch Satellite Images")

if fetch:
    st.success("Fetching Satellite Images...")

    try:
        reference_image = get_satellite_image(
            latitude, longitude, str(ref_start), str(ref_end)
        )
        comparison_image = get_satellite_image(
            latitude, longitude, str(comp_start), str(comp_end)
        )
    except ValueError as e:
        reference_image = None
        comparison_image = None
        st.error(f"⚠️ {e}")

    if reference_image is not None and comparison_image is not None:
        st.success("✅ Both Satellite Images Retrieved Successfully!")

        ref_thumbnail = get_image_thumbnail(reference_image)
        comp_thumbnail = get_image_thumbnail(comparison_image)
        ref_path = save_temp_image(ref_thumbnail)
        comp_path = save_temp_image(comp_thumbnail)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🛰️ Reference Image (T1)")
            st.image(ref_thumbnail, use_container_width=True)

        with col2:
            st.subheader("🛰️ Comparison Image (T2)")
            st.image(comp_thumbnail, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 TinyCD Change Detection")

        with st.spinner("Running TinyCD inference..."):
            result = predict_change(model, ref_path, comp_path, threshold=threshold)

        st.metric("Changed Area", f"{result['changed_fraction'] * 100:.2f}%")

        tab1, tab2, tab3 = st.tabs(["Binary Change Mask", "Probability Map", "Confidence Map"])

        with tab1:
            st.image(
                result["binary_mask_rgb"],
                caption=f"Binary change mask (threshold={threshold})  —  white = changed",
                use_container_width=True
            )

        with tab2:
            st.image(
                result["prob_map_rgb"],
                caption="Raw TinyCD probability output — blue = low change probability, red = high",
                use_container_width=True
            )

        with tab3:
            st.image(
                result["confidence_rgb"],
                caption="Model confidence — bright = model is sure either way, dark = near the decision boundary",
                use_container_width=True
            )

        with st.expander("🐛 Debug info"):
            st.write(f"Reference image (temp): `{ref_path}`")
            st.write(f"Comparison image (temp): `{comp_path}`")
            st.write(f"Probability map shape: {result['probability_map'].shape}")
            st.write(f"Probability range: [{result['probability_map'].min():.4f}, "
                      f"{result['probability_map'].max():.4f}]")
            st.write(f"Threshold used: {threshold}")
            st.write(f"Changed pixels: {int(result['binary_mask'].sum())} / "
                      f"{result['binary_mask'].size}")
            st.caption("Full preprocessing/model logs are also printed to the terminal running `streamlit run app.py`.")

    elif reference_image is not None or comparison_image is not None:
        st.error("One or both satellite images could not be retrieved.")
