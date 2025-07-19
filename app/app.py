import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from torchvision import transforms
import io

# -----------------------------
# 🔧 Load model + processor
# -----------------------------
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("vit-gpt2-captioning")  # Replace with your own path if local
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, processor, tokenizer

model, processor, tokenizer = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# 🔄 Image preprocessing
# -----------------------------
def preprocess_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    return processor(images=image, return_tensors="pt").pixel_values.to(device)

# -----------------------------
# 🧠 Generate caption
# -----------------------------
def generate_caption(image_tensor):
    output_ids = model.generate(image_tensor, max_length=50, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# -----------------------------
# 🖼️ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Image Captioning using ViT + GPT2")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        image_tensor = preprocess_image(image)
        caption = generate_caption(image_tensor)
        st.success("Caption generated!")
        st.markdown(f"**📝 Caption:** _{caption}_")
