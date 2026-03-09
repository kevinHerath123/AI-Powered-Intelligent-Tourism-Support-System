# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from PIL import Image
from classifier import get_prediction  # Only import what you need

st.set_page_config(page_title="Sri Lanka Landmark Recognition", page_icon="🏛️", layout="centered")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%); }
    .result-box { background: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Sri Lanka Landmark Recognition")
st.write("Upload a photo to identify the landmark and its location.")

# ✅ Simple model load check (no caching needed)
try:
    from classifier import init_classifier
    init_classifier()  # Just call it once
    st.sidebar.success("✅ Model Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Failed: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        result = get_prediction(image)  # ✅ Now works reliably

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(f"**🏛️ Landmark:** {result['name'].strip()}")
    st.write(f"**📍 Location:** {result['place'].strip()}")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👆 Upload an image to get started")

st.markdown("---")
st.caption("AI-Powered Intelligent Tourism Support System")