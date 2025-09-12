# TerraQuest AI: A Terrain Classification & Gear Recommendation App

This is a Streamlit web application that performs terrain classification from either ground-level images or satellite patches. It recommends appropriate gear (shoes or tyres) based on terrain types like `rocky`, `muddy`, `sandy`, etc.

---

## Features

- Classify terrains using satellite images via Earth Engine
- Accept and classify ground-level images uploaded by the user
- Supports two separate CNN models for satellite and ground input types
- Hosted on Streamlit Cloud
- Uses Google Earth Engine service account credentials securely
- Downloads:
  - Trained PyTorch models from Hugging Face
  - Dataset ZIP archive and extracts it dynamically in the app

---

## Supported Input Types

| Input Type | Source             | CNN Model Path                     |
|------------|--------------------|------------------------------------|
| Ground     | User-uploaded      | `terrain_classifier_ground.pth`    |
| Satellite  | Earth Engine Patch | `terrain_classifier_satellite.pth` |
