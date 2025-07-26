from terrain_classification_model import get_dataloaders, predict_image
from satellite_image_extractor import authenticate_earth_engine, get_satellite_patch
from weather_heuristics import Weather
import cv2
import json
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

# -------------------- Load Custom CSS --------------------

with open('style.css') as f:
    st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------- Main App --------------------
def main():
    st.title("🧭 Terrain Gear Recommender")
    st.markdown("---")

    catalog = load_catalog("gear_catalog.json")
    if not catalog:
        st.warning("No gear catalog found. Please check the file.")
        return

    input_type = st.radio("Choose Input Type:", ["Ground Image", "Coordinates"])

    prediction = ""
    condition = None
    image_path = ""

    if input_type == "Ground Image":
        uploaded_image = st.file_uploader("Upload Ground-Level Image", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            image_path = "landscape.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.read())

            model_path = "terrain_classifier_ground.pth"
            data_dir = "Dataset/ground_images"
            _, _, class_names = get_dataloaders(data_dir)

            prediction, confidences = predict_image(image_path, model_path, class_names, input_type='ground')

            st.image(image_path, caption="Uploaded Image", width=300)

            formatted_prediction = " & ".join(word.capitalize() for word in prediction.split("&"))
            st.success(f"Predicted Terrain: {formatted_prediction}")

            sorted_confidences = sorted(confidences.items(), key=lambda x: float(x[1].replace('%', '')), reverse=True)
            st.caption("Confidence Scores - " + " | ".join(f"{cls.capitalize()}: {conf}" for cls, conf in sorted_confidences))

    elif input_type == "Coordinates":
        authenticate_earth_engine()

        st.subheader("Select Coordinates")

        coord_input_mode = st.radio("How would you like to input coordinates?", ["Enter manually", "Choose on map"])

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
            else:
                st.subheader("**Click on the map below to pick coordinates:**")

        # Button triggers everything below
        if lat is not None and lon is not None and st.button("Classify Terrain from Coordinates"):
            # Generate satellite patch
            image_path = "patch.png"
            get_satellite_patch(lat, lon, output_path=image_path)

            # Ensure the patch file exists before proceeding
            if os.path.exists(image_path):

                model_path = "terrain_classifier_satellite.pth"
                data_dir = "Dataset/satellite_images"
                _, _, class_names = get_dataloaders(data_dir)

                prediction, confidences = predict_image(image_path, model_path, class_names, input_type='satellite')

                # Weather integration
                with open("WeatherAPI_Key.txt", "r") as f:
                    API_KEY = f.read().strip()
                condition, location_weather = Weather(API_KEY).assess_terrain_condition(lat, lon)

                st.image(image_path, caption="Satellite Image", width=300)
                st.caption(f"Weather Summary - {location_weather}")

                formatted_prediction = " & ".join(word.capitalize() for word in prediction.split("&"))
                st.success(f"Predicted Terrain: {formatted_prediction}")

                sorted_confidences = sorted(confidences.items(), key=lambda x: float(x[1].replace('%', '')), reverse=True)
                st.caption("Confidence Scores - " + " | ".join(f"{cls.capitalize()}: {conf}" for cls, conf in sorted_confidences))
                
                st.info(f"Terrain Condition: {condition}")
            else:
                st.error("Satellite patch could not be saved. Please try again.")

    # -------------------- Recommendation Section --------------------
    if prediction:
        item_type = st.selectbox("Get recommendations for:", ["Footwear", "Tyres"])

        predicted_classes = prediction.split("&")

        # Weather heuristics for coordinate input
        if input_type == "Coordinates" and condition:
            if condition == "Mud Prone":
                for terrain in predicted_classes:
                    if terrain in ["rocky", "vegetated", "sandy", "urban"] and "muddy" not in predicted_classes:
                        predicted_classes.insert(0, "muddy")
            elif condition == "Snow Prone":
                for terrain in predicted_classes:
                    if terrain in ["rocky", "vegetated", "sandy", "urban"] and "snowy" not in predicted_classes:
                        predicted_classes.insert(0, "snowy")

        st.markdown(f"**Final Terrain Classes:** {', '.join((i.capitalize() for i in predicted_classes))}")

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
