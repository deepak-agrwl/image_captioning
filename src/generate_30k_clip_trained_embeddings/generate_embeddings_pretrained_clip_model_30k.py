import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel as TransformersCLIPModel, DistilBertModel, DistilBertConfig, DistilBertTokenizer
from PIL import Image
import timm
import albumentations as A
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import functools
from itertools import islice

# Configuration
class CFG:
    image_path = "../../resources/input/flickr30k/flickr30k_images/flickr30k_images"
    captions_path = "../train_clip_model/captions.csv"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Parallel processing settings
    num_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers to avoid memory issues
    batch_size = 32  # Process images/captions in batches
    
    # Custom model configuration (copied from the notebook)
    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200
    pretrained = True
    trainable = True
    temperature = 1.0
    size = 224
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1

cfg = CFG()

# Custom model classes (copied from the notebook)
class ImageEncoder(nn.Module):
    def __init__(self, model_name=cfg.model_name, pretrained=cfg.pretrained, trainable=cfg.trainable):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name=cfg.text_encoder_model, pretrained=cfg.pretrained, trainable=cfg.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=cfg.projection_dim, dropout=cfg.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(self, temperature=cfg.temperature, image_embedding=cfg.image_embedding, text_embedding=cfg.text_embedding):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def get_transforms(mode="valid"):
    return A.Compose([
        A.Resize(cfg.size, cfg.size, always_apply=True),
        A.Normalize(max_pixel_value=255.0, always_apply=True),
    ])

def load_and_preprocess_image(image_path, transforms):
    """Load and preprocess a single image for custom model"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms(image=image)['image']
    return torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)

def batch_generator(items, batch_size):
    """Generate batches from a list of items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

class EmbeddingGenerator:
    def __init__(self, model_type="pretrained_clip"):
        """
        Initialize the embedding generator
        model_type: "pretrained_clip" or "custom_model"
        """
        self.model_type = model_type
        self.device = cfg.device
        
        if model_type == "pretrained_clip":
            print("Loading pretrained CLIP model...")
            self.clip_model = TransformersCLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
        elif model_type == "custom_model":
            print("Loading custom trained CLIP model...")
            self.custom_model = CLIPModel().to(self.device)
            self.custom_model.load_state_dict(torch.load("best.pt", map_location=self.device))
            self.custom_model.eval()
            self.tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
            self.transforms = get_transforms()
        else:
            raise ValueError("model_type must be 'pretrained_clip' or 'custom_model'")

    def get_image_embedding(self, image_path):
        """Get image embedding based on the selected model type"""
        if self.model_type == "pretrained_clip":
            return self._get_image_clip_embedding(image_path)
        else:
            return self._get_image_custom_embedding(image_path)

    def get_text_embedding(self, text):
        """Get text embedding based on the selected model type"""
        if self.model_type == "pretrained_clip":
            return self._get_text_clip_embedding(text)
        else:
            return self._get_text_custom_embedding(text)

    def _get_image_clip_embedding(self, image_path):
        """Returns the CLIP embedding for an image."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            return outputs.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def _get_text_clip_embedding(self, text):
        """Returns the CLIP embedding for a text string."""
        try:
            inputs = self.clip_processor(text=[text], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**inputs)
            return outputs.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
            return None

    def _get_image_custom_embedding(self, image_path):
        """Returns the custom model embedding for an image."""
        try:
            image_tensor = load_and_preprocess_image(image_path, self.transforms).to(self.device)
            
            with torch.no_grad():
                image_features = self.custom_model.image_encoder(image_tensor)
                image_embeddings = self.custom_model.image_projection(image_features)
            
            return image_embeddings.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def _get_text_custom_embedding(self, text):
        """Returns the custom model embedding for a text string."""
        try:
            # Tokenize caption
            encoded_caption = self.tokenizer(
                [text], 
                padding=True, 
                truncation=True, 
                max_length=cfg.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded_caption['input_ids'].to(self.device)
            attention_mask = encoded_caption['attention_mask'].to(self.device)
            
            with torch.no_grad():
                text_features = self.custom_model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = self.custom_model.text_projection(text_features)
            
            return text_embeddings.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
            return None

    def process_image_batch(self, image_batch):
        """Process a batch of images"""
        results = []
        for image_file in image_batch:
            try:
                image_path = os.path.join(cfg.image_path, image_file)
                embedding = self.get_image_embedding(image_path)
                
                if embedding is not None:
                    results.append((image_file, embedding, True))
                else:
                    results.append((image_file, None, False))
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results.append((image_file, None, False))
        
        return results

    def process_caption_batch(self, caption_batch):
        """Process a batch of captions"""
        results = []
        for idx, row in caption_batch:
            try:
                image_name = row['image']
                caption_number = row['caption_number']
                caption = row['caption']

                if pd.isna(caption):
                    results.append((idx, image_name, caption_number, None, False))
                    continue
                    
                caption = str(caption)
                embedding = self.get_text_embedding(caption)
                
                if embedding is not None:
                    results.append((idx, image_name, caption_number, embedding, True))
                else:
                    results.append((idx, image_name, caption_number, None, False))
                    
            except Exception as e:
                print(f"Error processing caption {idx}: {e}")
                results.append((idx, image_name, caption_number, None, False))
        
        return results

def generate_image_embeddings(generator):
    """Generate embeddings for all images with parallel processing"""
    print(f"Generating image embeddings using {generator.model_type}...")
    
    # Create output directory for image embeddings
    image_embeddings_dir = Path("image_embeddings")
    image_embeddings_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(cfg.image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Generating embeddings for {len(image_files)} images using {cfg.num_workers} workers...")
    
    successful_count = 0
    failed_count = 0
    
    # Process images in batches with parallel processing
    if generator.model_type == "pretrained_clip":
        # For pretrained CLIP, use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
            # Create batches
            batches = list(batch_generator(image_files, cfg.batch_size))
            
            # Submit all batches
            future_to_batch = {executor.submit(generator.process_image_batch, batch): batch for batch in batches}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing image batches"):
                batch_results = future.result()
                
                for image_file, embedding, success in batch_results:
                    if success and embedding is not None:
                        # Save embedding as .pt file
                        output_path = image_embeddings_dir / f"{os.path.splitext(image_file)[0]}.pt"
                        torch.save(torch.tensor(embedding), output_path)
                        successful_count += 1
                    else:
                        failed_count += 1
                        print(f"Failed to generate embedding for {image_file}")
    else:
        # For custom model, process sequentially due to GPU memory constraints
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                image_path = os.path.join(cfg.image_path, image_file)
                embedding = generator.get_image_embedding(image_path)
                
                if embedding is not None:
                    # Save embedding as .pt file
                    output_path = image_embeddings_dir / f"{os.path.splitext(image_file)[0]}.pt"
                    torch.save(torch.tensor(embedding), output_path)
                    successful_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to generate embedding for {image_file}")
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                failed_count += 1
                continue
    
    print(f"Image embeddings completed: {successful_count} successful, {failed_count} failed")
    print(f"Image embeddings saved to {image_embeddings_dir}")

def generate_caption_embeddings(generator):
    """Generate embeddings for all captions with parallel processing"""
    print(f"Generating caption embeddings using {generator.model_type}...")
    
    # Create output directory for caption embeddings
    caption_embeddings_dir = Path("caption_embeddings")
    caption_embeddings_dir.mkdir(exist_ok=True)
    
    # Load captions from the processed captions.csv file
    print("Loading captions...")
    df = pd.read_csv(cfg.captions_path)
    # The processed captions.csv has columns: image, caption_number, caption, id
    
    print(f"Generating embeddings for {len(df)} captions using {cfg.num_workers} workers...")
    
    successful_count = 0
    failed_count = 0
    
    # Process captions in batches with parallel processing
    if generator.model_type == "pretrained_clip":
        # For pretrained CLIP, use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
            # Create batches of (index, row) tuples
            caption_data = [(idx, row) for idx, row in df.iterrows()]
            batches = list(batch_generator(caption_data, cfg.batch_size))
            
            # Submit all batches
            future_to_batch = {executor.submit(generator.process_caption_batch, batch): batch for batch in batches}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing caption batches"):
                batch_results = future.result()
                
                for idx, image_name, caption_number, embedding, success in batch_results:
                    if success and embedding is not None:
                        # Save embedding with naming format: image_name_caption_index.pt
                        image_name_without_ext = os.path.splitext(image_name)[0]
                        output_filename = f"{image_name_without_ext}_caption_{caption_number}.pt"
                        output_path = caption_embeddings_dir / output_filename
                        torch.save(torch.tensor(embedding), output_path)
                        successful_count += 1
                    else:
                        failed_count += 1
                        print(f"Failed to generate embedding for caption {idx}: {image_name} {caption_number}")
    else:
        # For custom model, process sequentially due to GPU memory constraints
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing captions"):
            try:
                image_name = row['image']
                caption_number = row['caption_number']
                caption = row['caption']

                if pd.isna(caption):
                    print(f"Skipping caption {idx} {image_name} {caption_number} {caption} due to NaN value.")
                    failed_count += 1
                    continue
                    
                caption = str(caption)
                
                # Generate text embedding
                embedding = generator.get_text_embedding(caption)
                
                if embedding is not None:
                    # Save embedding with naming format: image_name_caption_index.pt
                    image_name_without_ext = os.path.splitext(image_name)[0]
                    output_filename = f"{image_name_without_ext}_caption_{caption_number}.pt"
                    output_path = caption_embeddings_dir / output_filename
                    torch.save(torch.tensor(embedding), output_path)
                    successful_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to generate embedding for caption {idx}: {image_name} {caption_number}")
                    
            except Exception as e:
                print(f"Error processing caption {idx} {image_name} {caption_number} {caption}: {e}")
                failed_count += 1
                continue
    
    print(f"Caption embeddings completed: {successful_count} successful, {failed_count} failed")
    print(f"Caption embeddings saved to {caption_embeddings_dir}")

def main(model_type="pretrained_clip"):
    """
    Main function to generate embeddings
    model_type: "pretrained_clip" (default) or "custom_model"
    """
    print(f"Starting embedding generation using {model_type}...")
    print(f"Using device: {cfg.device}")
    print(f"Using {cfg.num_workers} workers for parallel processing")
    print(f"Batch size: {cfg.batch_size}")
    
    # Initialize the embedding generator
    generator = EmbeddingGenerator(model_type)
    
    # Generate image embeddings
    print("\n" + "="*50)
    print("GENERATING IMAGE EMBEDDINGS")
    print("="*50)
    generate_image_embeddings(generator)
    
    # Generate caption embeddings
    print("\n" + "="*50)
    print("GENERATING CAPTION EMBEDDINGS")
    print("="*50)
    generate_caption_embeddings(generator)
    
    print("\n" + "="*50)
    print("EMBEDDING GENERATION COMPLETE!")
    print("="*50)
    print(f"Image embeddings saved in: image_embeddings/")
    print(f"Caption embeddings saved in: caption_embeddings/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate embeddings using CLIP models')
    parser.add_argument('--model_type', type=str, default='pretrained_clip', 
                       choices=['pretrained_clip', 'custom_model'],
                       help='Type of model to use: pretrained_clip (default) or custom_model')
    parser.add_argument('--num_workers', type=int, default=cfg.num_workers,
                       help=f'Number of workers for parallel processing (default: {cfg.num_workers})')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size,
                       help=f'Batch size for processing (default: {cfg.batch_size})')
    
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    cfg.num_workers = args.num_workers
    cfg.batch_size = args.batch_size
    
    main(args.model_type) 