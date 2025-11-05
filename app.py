import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

st.title("🧠 MRI Brain Tumor Detection System")
st.write("Upload an MRI image to detect if there is a tumor and what type it is.")

# --- MODEL URL ---
MODEL_PATH = "https://huggingface.co/ynathani22/brain-tumor/resolve/main/model2.h5"

# --- DOWNLOAD AND LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model():
    st.write("✅ Model downloaded successfully!")
    response = requests.get(MODEL_PATH)
    response.raise_for_status()
    with open("model2.h5", "wb") as f:
        f.write(response.content)

    # Load model in compatibility mode (safe_mode=False allows legacy .h5)
    model = tf.keras.models.load_model("model2.h5", compile=False, safe_mode=False)
    return model


# Load once and cache
model = load_model()

# --- CLASS LABELS ---
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# --- IMAGE PREPROCESSING ---
def preprocess_image(image):
    image = image.resize((150, 150))  # adjust to your model’s input size
    img_array = np.array(image) / 255.0
    if len(img_array.shape) == 2:  # grayscale → RGB
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🩻 Uploaded MRI", use_container_width=True)

    if st.button("🔍 Predict Tumor Type"):
        with st.spinner("Analyzing MRI..."):
            try:
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)
                class_index = np.argmax(prediction)
                tumor_type = class_labels[class_index]
                confidence = np.max(prediction) * 100
                st.success(f"**Detected Tumor Type:** {tumor_type.capitalize()} ({confidence:.2f}% confidence)")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload an MRI image to begin detection.")









