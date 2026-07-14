import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="VehicleVision AI",
    page_icon="🚘",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "vehicle_classifier.keras"
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

with open("class_names.json", "r") as file:
    class_names = json.load(file)

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0f172a, #1e3a8a);}
.block-container {max-width: 750px;}
.header {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
}
.result {
    background: #166534;
    color: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    margin-top: 18px;
}
.stButton > button {
    width: 100%;
    height: 48px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🚘 VehicleVision AI</h1>
    <p>Upload an image to identify the vehicle type.</p>
</div>
""", unsafe_allow_html=True)

st.metric("Validation Accuracy", "97.5%")

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Vehicle"):
        resized_image = image.resize((224, 224))
        image_array = np.array(resized_image)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)[0]
        predicted_index = np.argmax(prediction)

        vehicle = class_names[predicted_index]
        confidence = prediction[predicted_index] * 100

        st.markdown(f"""
        <div class="result">
            <h2>{vehicle.title()}</h2>
            <p>Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)