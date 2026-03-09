# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from PIL import Image
from classifier import get_prediction

# Page config
st.set_page_config(
    page_title="🏛️ Sri Lanka Landmark Recognition",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Darker gradient background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
        min-height: 100vh;
    }

    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0.5rem;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #b8b8b8;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    /* Upload section - remove extra padding */
    .upload-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        margin-top: 0.5rem;
    }

    /* Remove padding above uploader label */
    .stFileUploader {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    .stFileUploader label {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    /* Footer - full width */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        text-align: center;
        color: #666666;
        font-size: 0.85rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        background: rgba(10, 10, 15, 0.95);
        margin-left: calc(-50vw + 50%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* Remove extra padding from containers */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HERO SECTION
# -----------------------------------------------------------------------------
st.markdown('<p class="main-title">🏛️ Sri Lanka Landmark Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover the beauty of Sri Lanka with AI-powered image recognition</p>',
            unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
try:
    from classifier import init_classifier

    init_classifier()
    st.sidebar.success("✅ Model Ready")
except Exception as e:
    st.sidebar.error(f"❌ Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload a landmark photo",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Your Upload", use_column_width=True)

    # Predict
    with st.spinner("🔍 AI is analyzing..."):
        result = get_prediction(image)

    # Results card
    st.markdown(f"""
    <div class="result-card">
        <h3 style="margin-top: 0;">✨ Prediction Result</h3>
        <p style="font-size: 1.1rem;"><strong>🏛️ Landmark:</strong><br>{result['name'].strip()}</p>
        <p style="font-size: 1.1rem;"><strong>📍 Location:</strong><br>{result['place'].strip()}</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Placeholder
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; color: #b8b8b8;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">📸</div>
        <h3 style="margin: 0.5rem 0;">Ready to Explore?</h3>
        <p style="margin: 0.5rem 0;">Upload a photo of a Sri Lankan landmark above to get started!</p>
        <p style="font-size: 0.85rem; color: #666; margin: 0.5rem 0;">
            💡 Tip: Clear, well-lit photos work best
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER - Full Width
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    <p style="margin: 0;">🇱🇰 <strong>AI-Powered Intelligent Tourism Support System</strong></p>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.75rem; opacity: 0.7;">Built with TensorFlow • Streamlit • EfficientNetB0</p>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.7rem; opacity: 0.5;">© 2026</p>
</div>
""", unsafe_allow_html=True)