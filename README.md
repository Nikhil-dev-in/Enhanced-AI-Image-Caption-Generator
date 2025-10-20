# Enhanced AI Image Caption Generator (BLIP + Streamlit)

Generate descriptive captions for images using the BLIP model from Salesforce.

## Features
- Upload an image or provide an image URL
- GPU-aware: auto-selects CUDA when available
- Tunable generation: beams, max tokens, min length, length penalty, early stopping
- Example images to try quickly
- Caption history with clear and CSV export

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Usage
1. Choose an input method: Upload, Image URL, or pick an example
2. (Optional) Adjust settings in the sidebar under "Generation Settings"
3. Click "Generate Caption" to produce a caption
4. Manage history from the sidebar and export CSV

## Model
- Model: `Salesforce/blip-image-captioning-base`
- Library: Hugging Face Transformers

## Contributing (Hacktoberfest Welcome!)
We welcome contributions of all sizes. Some ideas:

- Add alternative caption models (BLIP large, ViT-GPT2)
- Add multilingual translation of captions
- Improve UI/UX and accessibility
- Add Dockerfile and deployment guides
- Add unit tests and CI workflow

Please open an issue to discuss ideas or pick up a "good first issue".

### Development
- Follow standard Python formatting and linting
- Keep UI responsive and avoid blocking calls
- Test both CPU and GPU paths if possible

## License
MIT
