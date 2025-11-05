import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import requests
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")
st.title("🧠 MRI Brain Tumor Detection System")
st.write("Upload an MRI image to detect if there is a tumor and what type it is.")

# --- MODEL DOWNLOAD SETTINGS ---
MODEL_PATH = "model2.h5"
MODEL_URL = "https://huggingface.co/yashika2212/brain-tumor/resolve/main/model2.h5"  # ✅ Your Hugging Face link

# --- DOWNLOAD MODEL IF NOT ALREADY PRESENT ---
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Hugging Face... (this may take a minute)"):
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("✅ Model downloaded successfully!")

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- CLASS LABELS ---
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# --- IMAGE UPLOAD SECTION ---
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image... Please wait"):
            # Preprocess the image
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

# --- SIDEBAR INFORMATION ---
st.sidebar.title("ℹ️ About this App")
st.sidebar.info(
    """
    This application uses a Convolutional Neural Network (CNN) model 
    trained on MRI scans to detect **brain tumors** and identify their type.

    **Model Source:** Uploaded by Yashika  
    **Framework:** TensorFlow / Keras  
    **Interface:** Streamlit
    """
)



