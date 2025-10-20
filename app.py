import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor
@st.cache_resource(show_spinner=True)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Initialize caption history
if "captions" not in st.session_state:
    st.session_state.captions = []

# Page config
st.set_page_config(page_title="Enhanced AI Image Caption Generator", layout="wide")
st.title("🖼️ Enhanced AI Image Caption Generator")
st.write("Upload an image and let AI generate a descriptive caption.")

# Sidebar with model info
with st.sidebar:
    st.header("Model Info")
    st.markdown("**Model:** BLIP Image Captioning Base")
    st.markdown("**Source:** Salesforce")
    st.markdown("**Library:** Hugging Face Transformers")

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                inputs = processor(images=image, return_tensors="pt")
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                st.session_state.captions.append(caption)

            st.success("Caption Generated!")
            st.subheader("Generated Caption:")
            st.write(caption)

    except Exception as e:
        st.error(f"Error processing image: {e}")

# Caption history
if st.session_state.captions:
    st.markdown("---")
    st.subheader("📝 Caption History")
    for i, cap in enumerate(st.session_state.captions[::-1], 1):
        st.markdown(f"**{i}.** {cap}")
