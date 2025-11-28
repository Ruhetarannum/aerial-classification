# app.py
import io
from pathlib import Path
import sys

import streamlit as st
from PIL import Image

# Ensure src/ is on Python path if inference.py is inside src/
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from inference import DetectorClassifier

# Basic page setup
st.set_page_config(
    page_title="Aerial Classification",
    layout="centered",
)

# Title
st.markdown("<h1 style='text-align: center;'>Aerial Classification</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Drone vs Bird Classifier</h3>", unsafe_allow_html=True)
st.write("")

# Load classifier once
@st.cache_resource
def load_model():
    return DetectorClassifier(models_dir="models", device="cpu")

dc = load_model()

# File upload
uploaded = st.file_uploader(
    "Upload an image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input Image", use_column_width=True)
    except:
        st.error("Invalid image file.")
        st.stop()

    # Run classification
    if st.button("Run Classification"):
        with st.spinner("Running model..."):
            img_bytes = uploaded.getvalue()
            annotated, results = dc.detect_and_classify(img_bytes)

        # Show annotated result
        st.image(annotated, caption="Annotated Image", use_column_width=True)

        # Show prediction
        st.subheader("Prediction")
        st.json(results)

        # Download annotated image
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG")
        buf.seek(0)

        st.download_button(
            "Download Annotated Image",
            data=buf,
            file_name="classified.jpg",
            mime="image/jpeg"
        )

else:
    st.info("Upload an image to get started.")
