import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from pathlib import Path

# Configuration class (copied from the notebook)
class CFG:
    debug = False
    image_path = "../../resources/input/flickr30k/flickr30k_images/flickr30k_images"
    captions_path = "../train_clip_model/captions.csv"
    batch_size = 64
    num_workers = 0
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 2
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

# Model classes (copied from the notebook)
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
    """Load and preprocess a single image"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms(image=image)['image']
    return torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)

def generate_image_embeddings():
    """Generate embeddings for all images"""
    print("Loading model...")
    model = CLIPModel().to(cfg.device)
    model.load_state_dict(torch.load("best.pt", map_location=cfg.device))
    model.eval()
    
    # Create output directory for image embeddings
    image_embeddings_dir = Path("image_embeddings")
    image_embeddings_dir.mkdir(exist_ok=True)
    
    transforms = get_transforms()
    image_files = [f for f in os.listdir(cfg.image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Generating embeddings for {len(image_files)} images...")
    
    with torch.no_grad():
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                image_path = os.path.join(cfg.image_path, image_file)
                image_tensor = load_and_preprocess_image(image_path, transforms).to(cfg.device)
                
                # Get image features and embeddings
                image_features = model.image_encoder(image_tensor)
                image_embeddings = model.image_projection(image_features)
                
                # Save embedding
                output_path = image_embeddings_dir / f"{os.path.splitext(image_file)[0]}.pt"
                torch.save(image_embeddings.cpu(), output_path)
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
    
    print(f"Image embeddings saved to {image_embeddings_dir}")

def generate_caption_embeddings():
    """Generate embeddings for all captions"""
    print("Loading model and tokenizer...")
    model = CLIPModel().to(cfg.device)
    model.load_state_dict(torch.load("best.pt", map_location=cfg.device))
    model.eval()
    
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
    
    # Create output directory for caption embeddings
    caption_embeddings_dir = Path("caption_embeddings")
    caption_embeddings_dir.mkdir(exist_ok=True)
    
    # Load captions from the processed captions.csv file
    print("Loading captions...")
    df = pd.read_csv(cfg.captions_path)
    # The processed captions.csv has columns: image, caption_number, caption, id
    
    print(f"Generating embeddings for {len(df)} captions...")
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing captions"):
            try:
                image_name = row['image']
                caption_number = row['caption_number']
                caption = row['caption']

                if pd.isna(caption):
                    print(f"Skipping caption {idx} {image_name} {caption_number} {caption} due to NaN value.")
                    continue
                caption = str(caption)
                
                # Tokenize caption
                encoded_caption = tokenizer(
                    [caption], 
                    padding=True, 
                    truncation=True, 
                    max_length=cfg.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                input_ids = encoded_caption['input_ids'].to(cfg.device)
                attention_mask = encoded_caption['attention_mask'].to(cfg.device)
                
                # Get text features and embeddings
                text_features = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = model.text_projection(text_features)
                
                # Save embedding with naming format: image_name_caption_index.pt
                image_name_without_ext = os.path.splitext(image_name)[0]
                output_filename = f"{image_name_without_ext}_caption_{caption_number}.pt"
                output_path = caption_embeddings_dir / output_filename
                torch.save(text_embeddings.cpu(), output_path)
                
            except Exception as e:
                print(f"Error processing caption {idx} {image_name} {caption_number} {caption}: {e}")
                continue
    
    print(f"Caption embeddings saved to {caption_embeddings_dir}")

def main():
    print("Starting embedding generation...")
    
    # Generate image embeddings
    # print("\n" + "="*50)
    # print("GENERATING IMAGE EMBEDDINGS")
    # print("="*50)
    # generate_image_embeddings()
    
    # Generate caption embeddings
    print("\n" + "="*50)
    print("GENERATING CAPTION EMBEDDINGS")
    print("="*50)
    generate_caption_embeddings()
    
    print("\n" + "="*50)
    print("EMBEDDING GENERATION COMPLETE!")
    print("="*50)
    print(f"Image embeddings saved in: image_embeddings/")
    print(f"Caption embeddings saved in: caption_embeddings/")

if __name__ == "__main__":
    main() 