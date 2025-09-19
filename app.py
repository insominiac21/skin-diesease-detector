# app.py
import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")
st.title("🩺 Simple Skin Lesion Classifier (DenseNet)")

# ----- Edit these if your training used a different order or names -----
CLASS_LABELS = [
    "Benign keratosis-like lesions",   # bkl
    "Melanocytic nevi",                # nv
    "Dermatofibroma",                  # df
    "Melanoma",                        # mel
    "Vascular lesions",                # vasc
    "Basal cell carcinoma",            # bcc
    "Actinic keratoses"                # akiec
]

MODEL_FILENAME = "model.keras"  # change if your model file has another name

@st.cache_resource
def load_densenet_model(path=MODEL_FILENAME):
    full = os.path.join(os.getcwd(), path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Model file not found: {full}")
    # load saved Keras model (architecture + weights)
    model = load_model(full, compile=False)
    # optional: compile for metrics (not required for inference)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model

def infer_target_size_from_model(model, default=(224, 224)):
    try:
        inp = model.input_shape
        # input_shape often (None, height, width, channels)
        if isinstance(inp, (list, tuple)):
            if isinstance(inp[0], (list, tuple)):
                inp = inp[0]
        if isinstance(inp, tuple) and len(inp) >= 4:
            h, w = inp[1], inp[2]
            if isinstance(h, int) and isinstance(w, int):
                return (w, h)  # PIL expects (width, height)
    except Exception:
        pass
    return default

def preprocess_image(pil_img, target_size):
    pil_img = pil_img.convert("RGB")
    pil_img = ImageOps.fit(pil_img, target_size, Image.LANCZOS)
    arr = np.asarray(pil_img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, h, w, 3)
    arr = preprocess_input(arr)        # DenseNet / ImageNet style preprocessing
    return arr

# ---- Load model ----
try:
    model = load_densenet_model(MODEL_FILENAME)
    st.success("Model loaded.")
except Exception as e:
    st.error("Could not load model:")
    st.error(str(e))
    model = None

# Determine target size
if model is not None:
    target_size = infer_target_size_from_model(model, default=(224, 224))
else:
    target_size = (224, 224)

st.write("Model input size (used for resize):", target_size, "(width, height)")

# ---- File uploader and prediction ----
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file and model is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        # preprocess and predict
        x = preprocess_image(image, target_size=target_size)
        preds = model.predict(x)[0]  # shape (num_classes,)
        # safety: if model produces more/less classes than CLASS_LABELS length
        n = preds.shape[0]
        labels = CLASS_LABELS[:n] if len(CLASS_LABELS) >= n else [f"Class {i}" for i in range(n)]

        top_idx = np.argmax(preds)
        top_label = labels[top_idx]
        top_prob = float(preds[top_idx])

        st.markdown(f"### 🧠 Prediction: **{top_label}** — {top_prob*100:.2f}%")

        # show full probabilities
        probs = {labels[i]: float(preds[i]) for i in range(n)}
        st.subheader("All class probabilities")
        st.table(probs)

    except Exception as e:
        st.error("Error during prediction:")
        st.error(str(e))
elif uploaded_file and model is None:
    st.warning("Model not loaded — cannot predict. Check the model file and restart.")
else:
    st.info("Upload an image to get a prediction.")
