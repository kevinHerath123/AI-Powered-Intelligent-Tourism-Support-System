import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from PIL import Image
from classifier import init_classifier, get_prediction

st.set_page_config(page_title="Landmark Recognition", page_icon="🏛️")


@st.cache_resource
def load_model():
    return init_classifier()


try:
    classifier = load_model()
    st.sidebar.success("✅ Model Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Model Load Failed: {str(e)}")
    st.stop()

st.title("🏛️ Sri Lanka Landmark Recognition")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        image = Image.open(uploaded_file)
        result = get_prediction(image)

    st.success("✅ Prediction Complete!")
    st.write(f"**Landmark:** {result['name'].strip()}")
    st.write(f"**Location:** {result['place'].strip()}")
else:
    st.info("👆 Please upload an image to get started.")