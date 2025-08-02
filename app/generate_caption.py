import streamlit as st
from PIL import Image
import torchvision.transforms as T

# Global variables
# Dataset configurations
DATASET_CONFIGS = {
    'flickr8k': {
        'image_dir': "../../resources/input/flickr8k/Images",
        'captions_file': "../../resources/input/flickr8k/captions.txt",
        'image_col': 'image',
        'caption_col': 'caption',
        'file_format': 'txt'
    },
    'flickr30k': {
        'image_dir': "../../resources/input/flickr30k/flickr30k_images/flickr30k_images",
        'captions_file': "../../resources/input/flickr30k/flickr30k_images/captions.csv",
        'image_col': 'image',
        'caption_col': 'caption',
        'file_format': 'csv'
    }
}

def test_model(model, dataset, device, num_samples=5, dataset_type='flickr8k'):
    """Test the trained model on randomly sampled images."""
    model.eval()
    
    # Create a simple data loader for testing
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    # Get dataset configuration
    config = DATASET_CONFIGS[dataset_type]
    
    test_dataset = CustomDataset(
        root_dir=config['image_dir'],
        captions_file=config['captions_file'],
        transform=transforms,
        dataset_type=dataset_type
    )
    
    print(f"\nTesting model on {num_samples} randomly sampled images from {dataset_type}:")
    print("-" * 50)
    
    # Generate random indices for testing
    random.seed(42)  # For reproducible results
    dataset_size = len(test_dataset)
    random_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))

    for i, idx in enumerate(random_indices):
        img, _ = test_dataset[idx]
        img = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.encoder(img)
            caps = model.decoder.generate_caption(features, vocab=dataset.vocab)
            caption = ' '.join(caps)
            
            print(f"Image {i+1} (index {idx}): {caption}")
            show_image(test_dataset[idx][0], title=caption)
            plt.show()
            plt.close()
            

# print("\nTesting model...")
# test_model(model, dataset, device, num_samples=3, dataset_type=args.dataset_type if hasattr(args, 'dataset_type') else args.dataset)


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
        cmodel, processor, tokenizer = load_model()
        caption = generate_caption(image, model, processor, tokenizer)
        st.success("Caption generated!")
        st.markdown(f"**📝 Caption:** _{caption}_")