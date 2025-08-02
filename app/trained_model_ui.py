import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/image_caption_generation")))
from flickr_image_caption_with_pytorch_resnet_lstm import Vocabulary, EncoderViT, GPT2Decoder, EncoderDecoder

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def ui_load_trained_model(model_path, device):
    """UI-specific model loading function, with fallback for state_dict-only files."""
    import torch
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Try if it's a checkpoint dict with hyperparameters
        if isinstance(checkpoint, dict) and "hyperparameters" in checkpoint:
            hyperparams = checkpoint['hyperparameters']
            model = EncoderDecoder(
                custom_vocab=checkpoint['vocab'],
                embed_size=hyperparams['embed_size'],
                hidden_size=hyperparams['hidden_size'],
                vocab_size=hyperparams['vocab_size'],
                num_layers=hyperparams['num_layers'],
                decoder_type=hyperparams.get('decoder_type', 'lstm'),
                attention_dim=hyperparams.get('attention_dim', 256),
                encoder_type=hyperparams.get('encoder_type', 'vit') 
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            vocab = checkpoint.get('vocab', None)
            if vocab is None:
                print("WARNING: No vocab found in checkpoint. You must provide the correct vocabulary manually!")
            return model, vocab, checkpoint
        else:
            # Assume it's just a state_dict
            print("Loaded file is a plain state_dict (no hyperparameters). Model params must be hardcoded in UI.")
            print(f"state_dict keys: {list(checkpoint.keys())}")
            # TODO: Update these hyperparams as per your actual training settings
            embed_size = 512  # example default
            hidden_size = 1024  # example default
            vocab_size = 30522  # example default, likely incorrect - must match your training!
            num_layers = 3
            decoder_type = "gpt2"
            attention_dim = 256
            model = EncoderDecoder(
                embed_size=embed_size,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                decoder_type=decoder_type,
                attention_dim=attention_dim
            ).to(device)
            model.load_state_dict(checkpoint)
            print("Model loaded with hardcoded parameters. Vocab is not loaded and must be supplied!")
            return model, None, None
    except Exception as e:
        print("Error loading model:", e)
        raise e

# Load the model
model_path = 'models/gpt2_vit.pth'
model, vocab, checkpoint = ui_load_trained_model(model_path, device)

# Image preprocessing
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def generate_caption(image, model, vocab, device):
    image = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encoder(image)
        caps = model.decoder.generate_caption(features, vocab=vocab)
        caption = ' '.join(caps)
    return caption

# Streamlit UI
st.set_page_config(page_title="Trained Image Caption Generator", layout="centered")
st.title("🖼️ Image Captioning with Trained ViT + GPT2 Model")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        caption = generate_caption(image, model, vocab, device)
        st.success("Caption generated!")
        st.markdown(f"**📝 Caption:** _{caption}_")
