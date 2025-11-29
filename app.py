import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import requests
import os

# -------------------------------
MODEL_URL = "https://huggingface.co/rushi7338/diabetic-retinopathy-alexnet/resolve/main/alexnet_dr_best.keras"
MODEL_PATH = "alexnet_dr_best.keras"

# Download once if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        r = requests.get(MODEL_URL, allow_redirects=True)
        open(MODEL_PATH, 'wb').write(r.content)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_names = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

# CLAHE preprocessing
def apply_CLAHE(image):
    img = image.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, A, B))
    rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return rgb

def predict_retina(image):
    img = cv2.resize(image, (224,224))
    img = apply_CLAHE(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id] * 100
    return class_names[class_id], confidence

# Streamlit UI
st.title("👁️ Diabetic Retinopathy Detection App")
st.write("Upload a retina image to classify DR severity.")

uploaded_file = st.file_uploader("Upload retina image", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_retina(image_np)
    st.subheader("🔍 Prediction")
    st.write(f"**Class:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
