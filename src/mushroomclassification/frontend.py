import io
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict/")

st.set_page_config(
    page_title="Mushroom Safety Classifier",
    page_icon="🍄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

IMAGE_PATH = Path(__file__).parent / "mushrooms.png"
st.image(str(IMAGE_PATH), width=80)
st.title("Can I eat it?")
st.markdown("Drag and drop an image to classify if it's edible.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file:
    try:
        file_data = uploaded_file.read()
        image = Image.open(io.BytesIO(file_data)).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        api_buffer = io.BytesIO()
        image.save(api_buffer, format="JPEG")
        api_buffer.seek(0)

        with st.spinner("Classifying the image..."):
            response = requests.post(
                API_URL,
                files={"file": ("image.jpg", api_buffer, "image/jpeg")},
            )

        if response.status_code == 200:
            predictions = response.json()

            st.markdown("### Prediction Confidence:")
            data = {
                "Class": [cls.replace("_", " ").title() for cls in predictions.keys()],
                "Confidence": list(predictions.values()),
            }
            st.bar_chart(data, x="Class", y="Confidence")

        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
