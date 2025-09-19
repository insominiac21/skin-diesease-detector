# app.py
import os
import io
import traceback
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model as keras_load_model

# -------------------------
# Page config & small UI
# -------------------------
st.set_page_config(page_title="Skin Disease Prediction", layout="centered")
st.title("🩺 Skin Disease Prediction App — Robust Loader + Diagnostics")

st.markdown(
    """
This app attempts to load your `model.h5` (either a full saved model or a weights-only file).
It rebuilds the CNN architecture used in your notebook if needed and provides debugging output
to help find why predictions might always be the same.
"""
)

# -------------------------
# Class labels (full names)
# -------------------------
# IMPORTANT: The order must match the numerical labels used during training.
CLASS_LABELS = [
    "Benign keratosis-like lesions",   # bkl
    "Melanocytic nevi",                # nv
    "Dermatofibroma",                  # df
    "Melanoma",                        # mel
    "Vascular lesions",                # vasc
    "Basal cell carcinoma",            # bcc
    "Actinic keratoses"                # akiec
]

NUM_CLASSES = len(CLASS_LABELS)

# -------------------------
# Notebook architecture (recreated)
# -------------------------
def build_model(input_shape=(75, 100, 3), num_classes=NUM_CLASSES):
    """
    Recreates the Sequential CNN architecture used in the uploaded notebook.
    Input shape is (height, width, channels) -> (75, 100, 3)
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# -------------------------
# Robust model loader
# -------------------------
@st.cache_resource
def load_my_model(model_path='model.h5'):
    """
    Try to load a full SavedModel / Keras model first. If that fails,
    rebuild model with build_model() and load weights.
    Returns (model, load_mode_string).
    """
    full_path = os.path.join(os.getcwd(), model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found at: {full_path}")

    # 1) Try to load the full model (architecture + weights)
    try:
        m = keras_load_model(full_path, compile=False)
        m.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        return m, "full_model"
    except Exception as e_full:
        # Keep the exception to include later if fallback fails
        full_err = e_full
        # show message in UI logs (streamlit reruns will cache)
        st.write("Note: keras_load_model() failed with message:")
        st.code(traceback.format_exc(limit=10))

    # 2) Fallback: rebuild architecture and load weights
    try:
        m = build_model()
        m.load_weights(full_path)
        m.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        return m, "weights_only"
    except Exception as e_weights:
        # both attempts failed -> raise informative error
        raise RuntimeError(f"Tried load_model() -> error:\n{full_err}\n\nTried build_model()+load_weights() -> error:\n{e_weights}")

# -------------------------
# Image preprocessing helpers
# -------------------------
def preprocess_image_pil(image, target_size=(100, 75), method="scale_0_1"):
    """
    Preprocess PIL image:
    - target_size is (width, height) for PIL.ImageOps.fit
    - method: "scale_0_1" -> divide by 255,
              "imagenet"   -> use EfficientNet / ImageNet-style preprocess_input
    Returns float32 numpy array shaped (1, height, width, 3)
    """
    # Ensure RGB
    image = image.convert("RGB")
    image = ImageOps.fit(image, target_size, Image.LANCZOS)  # target_size=(width, height)
    arr = np.asarray(image).astype(np.float32)

    if method == "scale_0_1":
        arr = arr / 255.0
    elif method == "imagenet":
        # Use EfficientNet preprocessing if needed (this subtracts mean / scales appropriately)
        from tensorflow.keras.applications.efficientnet import preprocess_input
        arr = preprocess_input(arr)
    else:
        raise ValueError("Unknown preprocessing method: " + str(method))

    # Keras models expect shape (batch, height, width, channels)
    arr = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2])
    return arr

# -------------------------
# UI: settings and model loading
# -------------------------
st.sidebar.header("Settings / Diagnostics")
model_file = st.sidebar.text_input("Model filename", "model.h5")
preproc_choice = st.sidebar.selectbox(
    "Preprocessing",
    options=["scale_0_1", "imagenet"],
    index=0,
    help="Choose scaling method. Use 'imagenet' if the model was trained using EfficientNet preprocessing or ImageNet mean/std."
)
show_summary = st.sidebar.checkbox("Show model summary", value=True)
show_raw_vector = st.sidebar.checkbox("Show raw softmax vector", value=True)
show_all_probs_table = st.sidebar.checkbox("Show full probabilities table", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Working directory:")
st.sidebar.write(os.getcwd())
try:
    st.sidebar.write(os.listdir(os.getcwd()))
except Exception:
    pass

# Attempt to load model
model = None
load_mode = None
try:
    with st.spinner("Loading model..."):
        model, load_mode = load_my_model(model_file)
    st.success(f"Model loaded as: **{load_mode}**")
    if show_summary and model is not None:
        buf = io.StringIO()
        model.summary(print_fn=lambda s: buf.write(s + "\n"))
        st.subheader("Model summary (top portion)")
        st.code(buf.getvalue()[:3000])
except Exception as e:
    st.error("Could not load model. See details below.")
    st.error(str(e))
    st.code(traceback.format_exc(limit=20))

# -------------------------
# File uploader + prediction
# -------------------------
st.header("Upload image for prediction")
uploaded = st.file_uploader("Choose an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# Optional second upload for quick comparison
uploaded2 = st.file_uploader("Optional: upload a second image to compare predictions", type=["jpg", "jpeg", "png"], key="file2")

def predict_and_report(image_pil):
    if model is None:
        st.warning("No model loaded — cannot predict.")
        return None

    # Preprocess
    target_size = (100, 75)  # PIL expects (width, height)
    try:
        inp = preprocess_image_pil(image_pil, target_size=target_size, method=preproc_choice)
    except Exception as e:
        st.error("Preprocessing failed: " + str(e))
        return None

    # Show input diagnostics
    st.write("Input shape:", inp.shape)
    st.write("Input min, max:", float(inp.min()), float(inp.max()))
    # Run prediction
    preds = model.predict(inp)[0]  # (num_classes,)
    # numeric checks
    st.write("Sum of softmax probabilities:", float(np.sum(preds)))
    if show_raw_vector:
        st.subheader("Raw softmax vector (first 7 values)")
        st.code(np.array2string(preds, precision=6, separator=", "))
    # Top-3
    top_idx = preds.argsort()[-3:][::-1]
    top3 = [(CLASS_LABELS[i], float(preds[i])) for i in top_idx]
    # Show main result with full label text explicitly
    best_label, best_prob = top3[0]
    # Use st.markdown for full label visibility (no truncation)
    st.markdown(f"### 🧠 Prediction: **{best_label}** — {best_prob*100:.2f}%")
    st.write("Top-3 predictions:")
    for lbl, p in top3:
        st.write(f"- {lbl}: {p*100:.2f}%")

    if show_all_probs_table:
        st.subheader("All class probabilities")
        probs_dict = {CLASS_LABELS[i]: float(preds[i]) for i in range(len(CLASS_LABELS))}
        # Display as table (not truncated)
        st.table(probs_dict)

    return preds

# Handle first upload
if uploaded:
    try:
        image = Image.open(uploaded)
        st.image(image, caption="Image 1", use_container_width=True)
        preds1 = predict_and_report(image)
    except Exception as e:
        st.error("Could not open / predict on the uploaded image:")
        st.error(str(e))

# Handle second upload (comparison)
if uploaded2:
    try:
        image2 = Image.open(uploaded2)
        st.image(image2, caption="Image 2", use_container_width=True)
        preds2 = predict_and_report(image2)
        # Compare vectors if both present
        if uploaded and (preds1 is not None) and (preds2 is not None):
            close = np.allclose(preds1, preds2, atol=1e-6)
            st.write("Are the two prediction vectors (softmax) nearly identical?", close)
            if close:
                st.warning(
                    "The outputs are nearly identical for two different images. This often indicates a mismatch "
                    "in preprocessing or a model that is highly biased. Try switching 'Preprocessing' to 'imagenet' "
                    "or re-check the model/weights."
                )
    except Exception as e:
        st.error("Could not open / predict on the second uploaded image:")
        st.error(str(e))

# -------------------------
# Quick sanity tests & tips
# -------------------------
st.markdown("---")
st.subheader("Quick troubleshooting tips")
st.markdown(
    """
- If the app loads the model as `full_model` this means the `.h5` file contained the full architecture and weights — that is usually the simplest case.
- If it loads as `weights_only`, it used the rebuilt architecture above and loaded weights into it — ensure that this architecture exactly matches the one used during training.
- If the app predicts the same class for every image:
  * Try switching the **Preprocessing** setting in the sidebar (use `imagenet` if the model was trained with EfficientNet).
  * Check that `CLASS_LABELS` order exactly matches the label encoding / training generator mapping.
  * If `Sum of softmax probabilities` is not ~1.0, something is wrong with the model's final activation or prediction pipeline.
- If you still get wrong predictions, paste the `Model loaded as: ...` message and the top portion of `model.summary()` into our chat and I will help further.
"""
)

st.caption("If you want me to debug further, paste the 'Model loaded as:' message and the first ~40 lines of the model summary here.")
