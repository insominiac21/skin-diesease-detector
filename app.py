# app.py
import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")
st.title("🩺 Simple Skin Lesion Classifier")

# ---- Configure these ----
CLASS_LABELS = [
    "Benign keratosis-like lesions",   # bkl
    "Melanocytic nevi",                # nv
    "Dermatofibroma",                  # df
    "Melanoma",                        # mel
    "Vascular lesions",                # vasc
    "Basal cell carcinoma",            # bcc
    "Actinic keratoses"                # akiec
]
MODEL_FILENAME = "model.keras"

# ---- Load model ----
@st.cache_resource
def load_densenet_model(path=MODEL_FILENAME):
    full = os.path.join(os.getcwd(), path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Model file not found: {full}")
    # load Keras model (architecture + weights)
    model = load_model(full, compile=False)
    # compile not required for inference but OK to set optimizer/loss
    try:
        model.compile(optimizer="adam", loss="categorical_crossentropy")
    except Exception:
        # compile sometimes fails if custom losses; it's fine for inference
        pass
    return model

def infer_target_size_from_model(model, default=(224, 224)):
    """
    Return a (width, height) tuple for PIL resizing. Uses model.input_shape if available.
    """
    try:
        inp = model.input_shape
        # normalize nested cases
        if isinstance(inp, list):
            inp = inp[0]
        if isinstance(inp, tuple) and len(inp) >= 4:
            # tuple is (None, height, width, channels)
            h = inp[1]
            w = inp[2]
            if isinstance(w, int) and isinstance(h, int):
                return (w, h)
    except Exception:
        pass
    return default

def preprocess_image(pil_img, target_size):
    pil_img = pil_img.convert("RGB")
    pil_img = ImageOps.fit(pil_img, target_size, Image.LANCZOS)
    arr = np.asarray(pil_img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, h, w, 3)
    arr = preprocess_input(arr)
    return arr

# ---- Try to load model and show status ----
try:
    model = load_densenet_model(MODEL_FILENAME)
    
except Exception as e:
    st.error("Could not load model:")
    st.error(str(e))
    model = None

# ---- Determine resize target ----
if model is not None:
    target_size = infer_target_size_from_model(model, default=(224, 224))
else:
    target_size = (224, 224)
st.write("Image resize target (width, height):", target_size)

# ---- Uploader and prediction ----
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

def safe_predict(model, x):
    """
    Runs model.predict and normalizes the result to a 1D numpy probability vector.
    Returns (success_bool, preds_array_or_error_message)
    """
    try:
        raw = model.predict(x)
    except AttributeError as ae:
        # catch cases like "'NoneType' object has no attribute 'pop'" etc
        msg = str(ae)
        if "pop" in msg:
            return False, (
                "Model.predict raised AttributeError mentioning 'pop'. "
                "This often means the model object is invalid (None) or a custom object/graph mismatch. "
                "Ensure the model loaded correctly and is a Keras model instance. "
                "Full error: " + msg
            )
        return False, "AttributeError during prediction: " + msg
    except Exception as e:
        return False, "Exception during model.predict: " + str(e)

    # If predict returned None
    if raw is None:
        return False, "model.predict returned None. That suggests the model object is not a valid Keras model for inference."

    # Normalize various possible shapes
    arr = np.asarray(raw)
    if arr.size == 0:
        return False, "model.predict returned an empty array."

    # If shape is (1, n) -> take first row
    if arr.ndim == 2 and arr.shape[0] == 1:
        preds = arr[0]
    # If shape is (n,) -> already fine
    elif arr.ndim == 1:
        preds = arr
    # If batch of many -> use first example
    elif arr.ndim == 2 and arr.shape[0] > 1:
        preds = arr[0]
    else:
        # unexpected shape
        return False, f"Unexpected prediction shape: {arr.shape}. Received raw output: {type(raw)}"

    # Ensure it's probabilities (softmax) or at least numeric
    try:
        preds = preds.astype(float)
    except Exception:
        return False, "Could not convert model output to float array."

    # If sums to 0 (weird), warn
    s = float(np.sum(preds))
    if not np.isfinite(s) or s <= 0:
        return False, f"Predicted probabilities sum to {s}. This is invalid. Check model output and final activation."

    # normalize in case model didn't return softmax (just for display)
    preds = preds / s
    return True, preds

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        # Use new streamlit API: width='stretch' instead of use_container_width=True
        st.image(image, caption="Uploaded image", width="stretch")
    except Exception as e:
        st.error("Failed to open uploaded image: " + str(e))
        image = None

    if image is not None:
        if model is None:
            st.warning("Model not loaded — cannot run prediction.")
        else:
            try:
                x = preprocess_image(image, target_size)
            except Exception as e:
                st.error("Preprocessing failed: " + str(e))
                x = None

            if x is not None:
                ok, result = safe_predict(model, x)
                if not ok:
                    st.error("Error during prediction:")
                    st.error(str(result))
                else:
                    preds = result
                    n = preds.shape[0]
                    labels = CLASS_LABELS[:n] if len(CLASS_LABELS) >= n else [f"Class {i}" for i in range(n)]
                    top_idx = int(np.argmax(preds))
                    top_label = labels[top_idx]
                    top_prob = float(preds[top_idx])
                    st.markdown(f"### 🧠 Prediction: **{top_label}** — {top_prob*100:.2f}%")

                    probs = {labels[i]: float(preds[i]) for i in range(n)}
                    st.subheader("All class probabilities")
                    st.table(probs)
