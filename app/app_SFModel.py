# import streamlit as st
# from PIL import Image
# import torch
# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# from torchvision import transforms
# import io
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # -----------------------------
# # 🔧 Load model + processor
# # -----------------------------
# @st.cache_resource
# def load_model():
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     return model, processor

# model, tokenizer = load_model()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)


# # -----------------------------
# # 🔄 Image preprocessing
# # -----------------------------
# def preprocess_image(image: Image.Image):
#     if image.mode != "RGB":
#         image = image.convert(mode="RGB")
#     return processor(images=image, return_tensors="pt").pixel_values.to(device)

# # -----------------------------
# # 🧠 Generate caption
# # -----------------------------

# def generate_caption(image, model, processor, tokenizer):
#     if image.mode != "RGB":
#         image = image.convert(mode="RGB")

#     pixel_values = processor(images=image, return_tensors="pt").pixel_values

#     # Use greedy decoding or sampling (avoid beam search for GPT2)
#     output_ids = model.generate(
#         pixel_values,
#         max_length=50,
#         do_sample=True,       # optional for sampling
#         top_k=50,
#         top_p=0.95,
#         num_return_sequences=1
#     )

#     caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return caption


# # -----------------------------
# # 🖼️ Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="Image Caption Generator", layout="centered")
# st.title("🖼️ Image Captioning using ViT + GPT2")

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     with st.spinner("Generating caption..."):
#         image_tensor = preprocess_image(image)
#         model, processor = load_model()
#         caption = generate_caption(image, model, processor)
#         st.success("Caption generated!")
#         st.markdown(f"**📝 Caption:** _{caption}_")




























import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

st.title("🖼️ Image Caption Generator (BLIP)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        # Preprocess image and generate caption
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)

    st.success("✅ Caption generated!")
    st.markdown(f"📋 **Caption:** _{caption}_")