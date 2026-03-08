# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from PIL import Image
from classifier import init_classifier, get_prediction

# Page config
st.set_page_config(page_title="🏛️ Sri Lanka Landmark Recognition", page_icon="🏛️", layout="centered")

# Simple custom styling
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .result-box { background: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; }
    .confidence-high { color: #22c55e; font-weight: bold; }
    .confidence-med { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏛️ Sri Lanka Landmark Recognition")
st.write("Upload a photo to identify the landmark and its location.")


# Load model
@st.cache_resource
def load_model():
    return init_classifier()


try:
    classifier = load_model()
    st.sidebar.success("✅ Model Loaded")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("Analyzing..."):
        result = get_prediction(image)

    # Show results
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(f"**🏛️ Landmark:** {result['name'].strip()}")
    st.write(f"**📍 Location:** {result['place'].strip()}")

    # Simple confidence indicator
    conf = result['confidence']
    if conf >= 0.8:
        st.markdown(f"**🎯 Confidence:** <span class='confidence-high'>{conf:.1%}</span>", unsafe_allow_html=True)
    elif conf >= 0.6:
        st.markdown(f"**🎯 Confidence:** <span class='confidence-med'>{conf:.1%}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**🎯 Confidence:** <span class='confidence-low'>{conf:.1%}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Upload an image to get started")

# Simple footer
st.markdown("---")
st.caption("Built with TensorFlow & Streamlit • AI-Powered Tourism Support")