import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize BLIP processor and model
@st.cache_resource(show_spinner=True)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

st.set_page_config(page_title="Enhanced AI Image Caption Generator", layout="wide")
st.title("🖼️ Enhanced AI Image Caption Generator")
st.write("Upload an image and let AI generate a descriptive caption.")

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            # Prepare image
            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        st.success("Caption Generated!")
        st.subheader("Generated Caption:")
        st.write(caption)
