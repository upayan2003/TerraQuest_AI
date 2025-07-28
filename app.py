from terrain_classification_model import get_dataloaders, predict_image
from satellite_image_extractor import authenticate_earth_engine, get_satellite_patch
from weather_heuristics import Weather
import cv2
import json
import requests
import os
import folium
import streamlit as st
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# -------------------- Load Gear Catalog --------------------
@st.cache_data
def load_catalog(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading catalog: {e}")
        return {}

# -------------------- Download Model from Hugging Face --------------------
@st.cache_data(show_spinner="Downloading model...")
def download_model(hf_url: str, filename: str) -> str:
    if not os.path.exists(filename):
        response = requests.get(hf_url, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename

HF_MODEL_URLS = {
    "ground": "https://huggingface.co/upayan2003/TerrainClassifiers/resolve/main/terrain_classifier_ground.pth",
    "satellite": "https://huggingface.co/upayan2003/TerrainClassifiers/resolve/main/terrain_classifier_satellite.pth"
}

def get_model_path(model_type):
    filename = f"terrain_classifier_{model_type}.pth"
    hf_url = HF_MODEL_URLS[model_type]
    return download_model(hf_url, filename)

import os
import requests
import zipfile
import streamlit as st

# -------------------- Download and Extract Dataset --------------------
@st.cache_data(show_spinner="Downloading dataset...")
def download_dataset(hf_url: str, extract_to: str = "data") -> str:
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, "TerrainDataset.zip")

    # Download only if not already present
    if not os.path.exists(zip_path):
        response = requests.get(hf_url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    dataset_root = os.path.join(extract_to, "Dataset")
    
    # # Extract only if not already extracted
    # if not os.path.exists(dataset_root):
    #     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #         zip_ref.extractall(extract_to)

    return dataset_root
    
# -------------------- Load Custom CSS --------------------
with open('style.css') as f:
    st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------- Main App --------------------
def main():
    st.title("TerraQuest AI")
    st.caption("### AI-Powered Terrain Detection with Smart Gear Recommendations")
    st.markdown("---")

    catalog = load_catalog("gear_catalog.json")
    if not catalog:
        st.warning("No gear catalog found. Please check the file.")
        return

    input_type = st.radio("Choose Input Type:", ["Ground Image", "Coordinates"], horizontal=True)

    prediction = ""
    condition = None
    image_path = ""
    
    HF_DATASET_URL = "https://huggingface.co/datasets/upayan2003/TerrainDataset/resolve/main/TerrainClassification.zip"
    dataset_root = download_dataset(HF_DATASET_URL)

    if input_type == "Ground Image":
        uploaded_image = st.file_uploader("Upload Ground-Level Image", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            image_path = "landscape.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.read())

            model_path = get_model_path("ground")
            data_dir = os.path.join(dataset_root, "ground_images")
            _, _, class_names = get_dataloaders(data_dir)

            prediction, confidences = predict_image(image_path, model_path, class_names, input_type='ground')

            st.image(image_path, caption="Uploaded Image", width=300)

            formatted_prediction = " & ".join(word.capitalize() for word in prediction.split("&"))
            st.success(f"Predicted Terrain: {formatted_prediction}")

            sorted_confidences = sorted(confidences.items(), key=lambda x: float(x[1].replace('%', '')), reverse=True)
            st.caption("Confidence Scores - " + " | ".join(f"{cls.capitalize()}: {conf}" for cls, conf in sorted_confidences))

    elif input_type == "Coordinates":
        authenticate_earth_engine(st.secrets["gcp"]["gcloud_json"])

        st.subheader("Select Coordinates")

        coord_input_mode = st.radio("How would you like to input coordinates?", ["Enter manually", "Choose on map"], horizontal=True)

        lat, lon = None, None

        if coord_input_mode == "Enter manually":
            lat = st.number_input("Latitude", format="%.6f")
            lon = st.number_input("Longitude", format="%.6f")

        elif coord_input_mode == "Choose on map":
            st.subheader("**Click on the map below to pick coordinates:**")

            # Default center of the map (India)
            m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
            m.add_child(folium.LatLngPopup())

            # Get map interaction
            map_click = st_folium(m, height=400, returned_objects=["last_clicked"], key="map")

            if map_click and map_click["last_clicked"]:
                lat = map_click["last_clicked"]["lat"]
                lon = map_click["last_clicked"]["lng"]

                # Re-render map with marker
                m = folium.Map(
                    location=[lat, lon],
                    zoom_start=10,
                    control_scale=False,
                    zoom_control=False,
                    dragging=False,
                    scrollWheelZoom=False,
                    doubleClickZoom=False
                )
                folium.Marker([lat, lon], popup=f"Selected Location: ({lat:.6f}, {lon:.6f})").add_to(m)
                m.add_child(folium.LatLngPopup())

                st.subheader("Selected Location Preview")
                st_folium(m, height=400, key="map_with_marker")

                st.markdown(f"**Selected Coordinates**: {lat:.6f}, {lon:.6f}")

        # Button triggers everything below
        if lat is not None and lon is not None:
            if st.button("Classify Terrain from Coordinates"):
                image_path = "patch.png"
                _, nightlight = get_satellite_patch(lat, lon, output_path=image_path)

                if os.path.exists(image_path):
                    model_path = get_model_path("satellite")
                    data_dir = os.path.join(dataset_root, "satellite_images")
                    _, _, class_names = get_dataloaders(data_dir)

                    prediction, confidences = predict_image(image_path, model_path, class_names, input_type='satellite')

                    # Heuristic: Override rocky → urban if nightlight is more than 7
                    apply_heuristic_note = False
                    if prediction == "rocky" and nightlight > 7:
                        prediction = "urban"
                        apply_heuristic_note = True

                    # Weather integration
                    with open("WeatherAPI_Key.txt", "r") as f:
                        API_KEY = f.read().strip()
                    condition, location_weather = Weather(API_KEY).assess_terrain_condition(lat, lon)

                    # Store in session
                    st.session_state["image_path"] = image_path
                    st.session_state["prediction"] = prediction
                    st.session_state["confidences"] = confidences
                    st.session_state["apply_heuristic_note"] = apply_heuristic_note
                    st.session_state["condition"] = condition
                    st.session_state["weather_summary"] = location_weather

                else:
                    st.error("Satellite patch could not be saved. Please try again.")

    if "prediction" in st.session_state:
        image_path = st.session_state["image_path"]
        prediction = st.session_state["prediction"]
        confidences = st.session_state["confidences"]
        apply_heuristic_note = st.session_state.get("apply_heuristic_note", False)
        condition = st.session_state["condition"]
        location_weather = st.session_state["weather_summary"]

        st.image(image_path, caption="Satellite Image", width=300)
        st.caption(f"Weather Summary - {location_weather}")

        formatted_prediction = " & ".join(word.capitalize() for word in prediction.split("&"))
        st.success(f"Predicted Terrain: **{formatted_prediction}**")

        sorted_confidences = sorted(confidences.items(), key=lambda x: float(x[1].replace('%', '')), reverse=True)
        st.caption("Confidence Scores - " + " | ".join(f"{cls.capitalize()}: {conf}" for cls, conf in sorted_confidences))

        if apply_heuristic_note:
            st.markdown("**Note:** Terrain classified as **Urban** on the basis of nightlight.")

        st.info(f"Terrain Condition: **{condition}**")

        # User feedback and manual selection
        st.markdown("### Is the prediction correct?")
        user_feedback = st.radio("Please confirm:", ["Yes", "No"], horizontal=True)

        # Initial predicted classes
        predicted_classes = prediction.split("&")

        if user_feedback == "No":
            st.warning("You can manually select the correct terrain type.")
            manual_selection = st.multiselect(
                "Select the correct terrain class(es):",
                options=["Muddy", "Rocky", "Sandy", "Vegetated", "Urban", "Snowy"],
                default=[]
            )
            if manual_selection:
                predicted_classes = [cls.lower() for cls in manual_selection]

        # Weather heuristics
        elif condition and user_feedback == "Yes":
            if condition == "Mud Prone":
                if any(t in ["rocky", "vegetated", "sandy", "urban"] for t in predicted_classes) and "muddy" not in predicted_classes:
                    predicted_classes.insert(0, "muddy")
            elif condition == "Snow Prone":
                if any(t in ["rocky", "vegetated", "sandy", "urban"] for t in predicted_classes) and "snowy" not in predicted_classes:
                    predicted_classes.insert(0, "snowy")

        item_type = st.selectbox("Get recommendations for:", ["Footwear", "Tyres"])

        st.markdown(f"**Final Terrain Classes:** {', '.join((i.capitalize() for i in predicted_classes))}")

    # -------------------- Recommendation Section --------------------
        items = []
        seen = set()
        for terrain_key in predicted_classes:
            terrain_items = catalog.get(terrain_key, {}).get(item_type.lower(), [])
            for item in terrain_items:
                key = (item['brand'], item['model'])
                if key not in seen:
                    seen.add(key)
                    items.append(item)

        if items:
            st.subheader(f"Top {len(items)} recommended {item_type} for this terrain:")
            for i, item in enumerate(items, 1):
                st.markdown(f"**{i}. {item['brand']} - {item['model']}**")
                st.markdown(f"[View Item]({item['link']})")
        else:
            st.warning("No recommendations found for this terrain.")

if __name__ == "__main__":
    main()
