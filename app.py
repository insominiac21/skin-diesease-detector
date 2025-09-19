# ---------- Robust loader + notebook architecture (paste into your Streamlit app) ----------
import os
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model as keras_load_model

# Recreate the model architecture exactly as in your notebook
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
def load_my_model(model_path='model.h5'):
    full_path = os.path.join(os.getcwd(), model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found: {full_path}")

    # 1) Try to load the full model first (saved architecture + weights)
    try:
        m = keras_load_model(full_path, compile=False)
        # compile for completeness
        m.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        loaded_as = "full_model"
        return m, loaded_as
    except Exception as e_full:
        # show the failure in logs then try weights-only fallback
        st.write("load_model() failed:", str(e_full))

    # 2) Fallback: build architecture in code and load weights
    try:
        m = build_model()   # uses the notebook architecture above
        m.load_weights(full_path)
        m.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        loaded_as = "weights_only"
        return m, loaded_as
    except Exception as e_weights:
        # If both fail, raise a combined, informative error
        raise RuntimeError(f"Tried load_model() -> error: {e_full}\nTried build_model()+load_weights() -> error: {e_weights}")

# helper to show model summary truncated
def show_model_summary(m):
    buf = io.StringIO()
    m.summary(print_fn=lambda s: buf.write(s + "\n"))
    summary_str = buf.getvalue()
    # show first 3000 chars to avoid flooding UI
    st.code(summary_str[:3000])

# ---------- End loader ----------
