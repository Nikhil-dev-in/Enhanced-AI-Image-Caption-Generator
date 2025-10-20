import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO, StringIO
import csv
from datetime import datetime

# Load BLIP model and processor
@st.cache_resource(show_spinner=True)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Initialize caption history
if "captions" not in st.session_state:
    st.session_state.captions = []
if "captions_records" not in st.session_state:
    st.session_state.captions_records = []

# Page config
st.set_page_config(page_title="Enhanced AI Image Caption Generator", layout="wide")
st.title("🖼️ Enhanced AI Image Caption Generator")
st.write("Upload an image or provide a URL; configure generation and let AI caption it.")

# Sidebar with model info
with st.sidebar:
    st.header("Model Info")
    st.markdown("**Model:** BLIP Image Captioning Base")
    st.markdown("**Source:** Salesforce")
    st.markdown("**Library:** Hugging Face Transformers")
    st.markdown(f"**Device:** {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")

    st.markdown("---")
    st.header("Generation Settings")
    max_new_tokens = st.slider("Max new tokens", min_value=8, max_value=64, value=32, step=1)
    num_beams = st.slider("Beam search beams", min_value=1, max_value=8, value=3, step=1)
    min_length = st.slider("Min length", min_value=1, max_value=16, value=1, step=1)
    length_penalty = st.slider("Length penalty", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    early_stopping = st.checkbox("Early stopping", value=True)

    st.markdown("---")
    if st.session_state.captions:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear History"):
                st.session_state.captions = []
                st.session_state.captions_records = []
        with col_b:
            # Prepare CSV for download
            csv_io = StringIO()
            writer = csv.writer(csv_io)
            writer.writerow(["#", "timestamp_iso", "caption"])
            for idx, rec in enumerate(st.session_state.captions_records, 1):
                writer.writerow([idx, rec["timestamp"], rec["caption"]])
            st.download_button(
                label="Download Captions CSV",
                data=csv_io.getvalue(),
                file_name="captions.csv",
                mime="text/csv",
                use_container_width=True,
            )

tabs = st.tabs(["Upload Image", "Image URL", "Examples"])

with tabs[0]:
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            image = None
        if image is not None and st.button("Generate Caption", use_container_width=True):
            with st.spinner("Generating caption..."):
                inputs = processor(images=image, return_tensors="pt").to(device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    min_length=min_length,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                )
                caption = processor.decode(out[0], skip_special_tokens=True)
                st.session_state.captions.append(caption)
                st.session_state.captions_records.append({
                    "caption": caption,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            st.success("Caption Generated!")
            st.subheader("Generated Caption:")
            st.write(caption)

with tabs[1]:
    url = st.text_input("Paste image URL (png/jpg/jpeg)")
    col1, col2 = st.columns([1,1])
    with col1:
        fetch_clicked = st.button("Fetch Image", use_container_width=True)
    with col2:
        gen_from_url = st.button("Generate from URL", use_container_width=True)

    url_image = None
    if fetch_clicked and url:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            url_image = Image.open(BytesIO(resp.content)).convert("RGB")
            st.image(url_image, caption="Fetched Image", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to fetch image: {e}")

    if gen_from_url and url:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
            st.image(image, caption="Fetched Image", use_column_width=True)
            with st.spinner("Generating caption..."):
                inputs = processor(images=image, return_tensors="pt").to(device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    min_length=min_length,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                )
                caption = processor.decode(out[0], skip_special_tokens=True)
                st.session_state.captions.append(caption)
                st.session_state.captions_records.append({
                    "caption": caption,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            st.success("Caption Generated!")
            st.subheader("Generated Caption:")
            st.write(caption)
        except Exception as e:
            st.error(f"Failed to generate from URL: {e}")

with tabs[2]:
    st.write("Try one of these example images:")
    example_urls = [
        "https://images.unsplash.com/photo-1504208434309-cb69f4fe52b0",
        "https://images.unsplash.com/photo-1519681393784-d120267933ba",
        "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee",
    ]
    cols = st.columns(len(example_urls))
    for i, ex_url in enumerate(example_urls):
        with cols[i]:
            st.markdown(f"[{ex_url}]({ex_url})")
            if st.button("Caption", key=f"ex_{i}"):
                try:
                    resp = requests.get(ex_url, timeout=15)
                    resp.raise_for_status()
                    image = Image.open(BytesIO(resp.content)).convert("RGB")
                    st.image(image, caption="Example Image", use_column_width=True)
                    with st.spinner("Generating caption..."):
                        inputs = processor(images=image, return_tensors="pt").to(device)
                        out = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            num_beams=num_beams,
                            min_length=min_length,
                            length_penalty=length_penalty,
                            early_stopping=early_stopping,
                        )
                        caption = processor.decode(out[0], skip_special_tokens=True)
                        st.session_state.captions.append(caption)
                        st.session_state.captions_records.append({
                            "caption": caption,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    st.success("Caption Generated!")
                    st.subheader("Generated Caption:")
                    st.write(caption)
                except Exception as e:
                    st.error(f"Failed to caption example: {e}")

# Caption history
if st.session_state.captions:
    st.markdown("---")
    st.subheader("📝 Caption History")
    for i, cap in enumerate(st.session_state.captions[::-1], 1):
        st.markdown(f"**{i}.** {cap}")
