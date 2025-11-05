import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import requests
import os

# --- Page setup ---
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")
st.title("🧠 MRI Brain Tumor Detection System")
st.write("Upload an MRI image to detect if there is a tumor and what type it is.")

# --- Model download from Google Drive ---
MODEL_PATH = "model2.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1aaK3MpR3RDzIrqMmbTkO4KvtriMV9goB"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait (~120MB)"):
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("✅ Model downloaded successfully!")

# --- Load model (cached so it loads once) ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- Class labels ---
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# --- Image upload section ---
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing... Please wait"):
            # Preprocess image
            IMAGE_SIZE = 128
            img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence_score = np.max(predictions, axis=1)[0]
            result = class_labels[predicted_class_index]

            # Display result
            if result == "notumor":
                st.success(f"🧠 **No Tumor Detected**\nConfidence: {confidence_score*100:.2f}%")
            else:
                st.error(f"⚠️ **Tumor Detected: {result.capitalize()}**\nConfidence: {confidence_score*100:.2f}%")

