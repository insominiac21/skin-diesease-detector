# ----- DEBUG + ROBUST MODEL LOADER -----
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.optimizers import Adam

@st.cache_resource
def load_my_model(model_path='model.h5'):
    full_path = os.path.join(os.getcwd(), model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found: {full_path}")

    # First try to load the full saved model (architecture + weights)
    try:
        m = keras_load_model(full_path)
        # compile for completeness
        m.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        loaded_as = "full_model"
    except Exception as e_full:
        # If that fails, fallback to building the architecture and loading weights
        try:
            m = build_model()   # your build_model() from the notebook architecture
            m.load_weights(full_path)
            m.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
            loaded_as = "weights_only"
        except Exception as e_weights:
            # raise a combined error that includes both attempts
            raise RuntimeError(f"Tried load_model() -> error: {e_full}\nTried build_model()+load_weights() -> error: {e_weights}")

    return m, loaded_as

# Use the loader
try:
    model, load_mode = load_my_model('model.h5')
    st.success(f"Model loaded ({load_mode}).")
    # show model summary (collapsing long text)
    import io, sys
    buf = io.StringIO()
    model.summary(print_fn=lambda s: buf.write(s + "\n"))
    summary_str = buf.getvalue()
    st.text("Model summary (truncated):")
    st.code(summary_str[:2000])  # show first chunk; increase if needed
except Exception as e:
    st.error(f"Could not load model: {e}")
    model = None

# ----- Prediction diagnostics (replace your existing prediction section) -----
def debug_predict(image, show_debug=True):
    # Preprocess same as before
    input_data = preprocess_image(image).astype(np.float32)
    st.write("Input shape (batch, h, w, c):", input_data.shape)
    st.write("Input min/max:", float(input_data.min()), float(input_data.max()))

    preds = model.predict(input_data)[0]  # (num_classes,)
    st.write("Raw softmax vector:", preds.tolist())
    st.write("Sum of probs (should be 1.0):", float(preds.sum()))

    # top-3
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [(CLASS_LABELS[i], float(preds[i])) for i in top3_idx]
    st.write("Top-3 predictions:", top3)

    # quick check if predictions are extremely skewed
    if preds.max() > 0.99:
        st.warning("Model is highly confident (>99%) for a single class — this can be normal for some images but may indicate a problem if it's the case for all inputs.")

    return preds, top3

# Handling single or multiple uploads
uploaded_file = st.file_uploader("Upload an image of a skin lesion", type=["jpg","jpeg","png"])
if uploaded_file and model is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded", use_container_width=True)
    preds, top3 = debug_predict(img)

    st.success(f"Predicted: **{top3[0][0]}** ({top3[0][1]*100:.2f}%)")
    # Show full probability table
    st.subheader("All class probabilities")
    probs = {CLASS_LABELS[i]: float(preds[i]) for i in range(len(CLASS_LABELS))}
    st.table(probs)

    # OPTIONAL: if user uploads a second image, compare preds quickly
    st.info("If you want, upload a second image now to compare predictions.")
    uploaded_file2 = st.file_uploader("Upload a second image (optional) to compare", type=["jpg","jpeg","png"], key="u2")
    if uploaded_file2:
        img2 = Image.open(uploaded_file2)
        st.image(img2, caption="Second image", use_container_width=True)
        preds2, top3_2 = debug_predict(img2)
        same = np.allclose(preds, preds2, atol=1e-6)
        st.write("Are the two softmax outputs identical (within tolerance)?", same)
else:
    if model is None:
        st.warning("Model not loaded. Fix load errors above.")
    else:
        st.info("Upload an image to test.")
