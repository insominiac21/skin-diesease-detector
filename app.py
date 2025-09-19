
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageOps
import numpy as np
import os

st.set_page_config(page_title="Skin Disease Prediction", layout="centered")

st.title("🩺 Skin Disease Prediction App (Updated architecture)")



# Rebuild the CNN architecture from the uploaded notebook
def build_model(input_shape=(75, 100, 3), num_classes=7):
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

@st.cache_resource
def load_model(model_path='model.h5'):
    # Build and load weights (weights expected to match this architecture)
    model = build_model()
    full_path = os.path.join(os.getcwd(), model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model weights not found at: {full_path}")
    model.load_weights(full_path)
    # compile is optional for inference but kept for completeness
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Try loading the model (wrapped to show friendly error in UI)
try:
    model = load_model('model.h5')
    
except Exception as e:
    st.error(f"Could not load model: {e}")
    model = None

# Preprocess uploaded image to match notebook input_shape (75,100,3)
def preprocess_image(image):
    # Notebook used input_shape = (75,100,3) -> PIL expects size as (width, height)
    target_size = (100, 75)
    image = image.convert("RGB")
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0
    # model expects shape (batch, height, width, channels) -> (1,75,100,3)
    image_array = image_array.reshape(1, 75, 100, 3)
    return image_array

# Full human-readable class names (corresponding to the model's label ordering used previously)
CLASS_LABELS = [
    "Benign keratosis-like lesions",   # bkl
    "Melanocytic nevi",                # nv
    "Dermatofibroma",                  # df
    "Melanoma",                        # mel
    "Vascular lesions",                # vasc
    "Basal cell carcinoma",            # bcc
    "Actinic keratoses"                # akiec
]

uploaded_file = st.file_uploader("Upload an image of a skin lesion", type=["jpg", "jpeg", "png"])

if uploaded_file and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("🔍 Classifying...")

    try:
        input_data = preprocess_image(image)
        preds = model.predict(input_data)[0]  # shape (7,)
        top_idx = np.argmax(preds)
        top_label = CLASS_LABELS[top_idx]
        top_prob = float(preds[top_idx])

        # Show prediction and probabilities
        st.success(f"🧠 Prediction: **{top_label}** ({top_prob*100:.2f}%)")
        st.subheader("All class probabilities")
        probs = {CLASS_LABELS[i]: float(preds[i]) for i in range(len(CLASS_LABELS))}
        st.table(probs)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
elif uploaded_file and model is None:
    st.warning("Model failed to load — cannot make predictions. Check the working directory contents and model file path.")
else:
    st.info("Upload an image to get started.")
