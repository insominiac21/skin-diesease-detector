import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# 🔍 Debug: Show current directory contents
st.write("📁 Files in working directory:", os.listdir(os.getcwd()))

# ✅ Load full model (architecture + weights)
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'model.h5')
    return tf.keras.models.load_model(model_path)

model = load_model()

# 🧼 Preprocess uploaded image
def preprocess_image(image):
    target_size = (224, 224)
    image = image.convert("RGB")  # Ensure 3 channels
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0
    return image_array.reshape(1, 224, 224, 3)

# 🎯 Streamlit UI
st.title("🩺 Skin Disease Prediction App")

uploaded_file = st.file_uploader("Upload an image of a skin lesion", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("🔍 Classifying...")

    input_data = preprocess_image(image)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)

    # 🏷️ Replace with your actual class labels
    class_labels = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
    st.success(f"🧠 Prediction: **{class_labels[predicted_class]}**")
