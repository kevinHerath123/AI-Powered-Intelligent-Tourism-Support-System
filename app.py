# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from PIL import Image
from classifier import get_prediction

# Page config
st.set_page_config(
    page_title=" Sri Lanka Landmark Recognition",
    page_icon="📍",
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

    /* Main title - visible at top, larger font */
    .main-title {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin: 0.4rem 0 0.3rem 0 !important;
    }

    /* Subtitle - larger font */
    .subtitle {
        text-align: center;
        color: #c8c8c8;
        font-size: 1.5rem;
        margin: 0 !important;
        padding: 0 !important;
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

    /* Upload section - NO PADDING */
    .upload-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 0 !important;
        margin: 0.5rem 0 !important;
    }

    /* Remove ALL padding/margin from uploader */
    .stFileUploader {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    .stFileUploader label {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    [data-testid="stFileUploader"] {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    [data-testid="stFileUploader"] label {
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

    /* Minimize block container padding */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 4rem !important;
    }

    /* Remove all element margins */
    .element-container {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    /* Remove section margins */
    section {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HERO SECTION - NO EXTRA SPACING
# -----------------------------------------------------------------------------
st.markdown('<p class="main-title">📍 Sri Lanka Landmark Recognition</p>', unsafe_allow_html=True)
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
# MAIN CONTENT - NO PADDING
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
    with st.spinner("🔍 Analyzing picture..."):
        result = get_prediction(image)

    # Results card
    st.markdown(f"""
    <div class="result-card">
        <h3 style="margin-top: 0;">✨ Prediction Result</h3>
        <p style="font-size: 1.2rem;"><strong>📌 Landmark:</strong><br>{result['name'].strip()}</p>
        <p style="font-size: 1.2rem;"><strong>📍 Location:</strong><br>{result['place'].strip()}</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Placeholder
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #b8b8b8;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">📸</div>
        <h3 style="margin: 0.5rem 0; font-size: 1.5rem;">Ready to Explore?</h3>
        <p style="margin: 0.5rem 0; font-size: 1.1rem;">Upload a photo of a Sri Lankan landmark above to get started!</p>
        <p style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">
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
    <p style="margin: 0;">🇱🇰 AI-Powered Intelligent Tourism Support System</p>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.7rem; opacity: 0.5;">© 2026</p>
</div>
""", unsafe_allow_html=True)