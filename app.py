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
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
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
        margin-bottom: 2rem;
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

    /* Upload section */
    .upload-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin: 1.5rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888888;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
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
    <div style="text-align: center; padding: 2rem; color: #b8b8b8;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📸</div>
        <h3>Ready to Explore?</h3>
        <p>Upload a photo of a Sri Lankan landmark above to get started!</p>
        <p style="font-size: 0.9rem; color: #888;">
            💡 Tip: Clear, well-lit photos work best
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    <p>🇱🇰 <strong>AI-Powered Intelligent Tourism Support System</strong></p>
    <p>Built with TensorFlow • Streamlit • EfficientNetB0</p>
    <p style="font-size: 0.8rem; opacity: 0.8;">© 2026</p>
</div>
""", unsafe_allow_html=True)