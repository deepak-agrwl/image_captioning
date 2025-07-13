import os
import sys
import numpy as np
import pandas as pd
import random
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from collections import Counter
import spacy
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as T
import torch.optim.lr_scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import json
import time
import argparse

# Additional imports for metrics and visualization (assuming these files exist)
from metrics_utils import compute_metrics
from attention_visualization import visualize_attention # Ensure this is updated for spatial attention
from experiment_utils import (
    select_reference_images, load_reference_indices, save_reference_indices,
    run_caption_comparison, plot_caption_comparison, evaluate_on_validation
)
from training_utils import save_model_checkpoint, save_loss_curves, save_metrics_history

# Global variables (paths will be set via argparse)
PRETRAINED_RESNET_MODEL_PATH = None
MODEL_SAVE_DIR = None

# Dataset configurations (as in your original script)
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

# Default dataset (can be overridden via command line)
DEFAULT_DATASET = 'flickr30k'

# Load spaCy model
try:
    spacy_eng = spacy.load('en_core_web_sm')
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    sys.exit(1)


class Vocabulary:
    """Vocabulary class for tokenizing and numericalizing text."""
    
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        """Tokenize text using spaCy."""
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        """Build vocabulary from list of sentences."""
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        """Convert text to numerical indices."""
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


class CustomDataset(Dataset):
    """Custom dataset for Flickr8k and Flickr30k images and captions."""
    
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5, dataset_type='flickr8k'):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_type = dataset_type
        
        if dataset_type == 'flickr8k':
            self.df = pd.read_csv(captions_file, sep=',')
        elif dataset_type == 'flickr30k':
            self.df = pd.read_csv(captions_file)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return img, torch.tensor(caption_vec)


class CapsCollate:
    """Custom collate function for batching."""
    
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets


class EncoderCNN(nn.Module):
    """
    CNN encoder using ResNet50, modified to output spatial features for attention.
    The output will be (batch_size, num_pixels, encoder_dim)
    """
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # Use recommended weights
        
        # Load pretrained weights (if a specific path is provided and exists)
        if PRETRAINED_RESNET_MODEL_PATH and os.path.exists(PRETRAINED_RESNET_MODEL_PATH):
            print(f"Loading pretrained ResNet50 weights from {PRETRAINED_RESNET_MODEL_PATH}")
            resnet.load_state_dict(torch.load(PRETRAINED_RESNET_MODEL_PATH, map_location='cpu'))
        else:
            print("Using ImageNet pretrained ResNet50 weights (default).")
        
        # Freeze parameters of early layers
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # We want features from layer4 before global average pooling to maintain spatial information
        # ResNet50's layer4 outputs 2048 channels.
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2]) # Remove avgpool and fc
        
        # A 1x1 convolution to reduce feature dimensions if desired, or keep 2048
        # Here, we map 2048 to embed_size, which will be the encoder_dim for attention
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # Keep for compatibility if needed, but not for spatial attention
        self.fc = nn.Linear(resnet.fc.in_features, embed_size) # This layer will be used if not using spatial attention

        # Define a linear layer to project the spatial features if embed_size is different from 2048
        # This will be the new 'embed' layer for spatial features
        self.spatial_feature_projection = nn.Linear(2048, embed_size) # ResNet50 layer4 output channels is 2048

    def forward(self, images):
        # Output from resnet_features will be (batch, 2048, H_prime, W_prime)
        # For 224x224 input, H_prime and W_prime are typically 7x7
        features = self.resnet_features(images) # (batch, 2048, 7, 7) for 224x224 input

        # Reshape to (batch, H_prime * W_prime, feature_dim) for attention
        batch_size, feature_dim, H, W = features.size()
        features = features.view(batch_size, feature_dim, H * W).permute(0, 2, 1) # (batch, H*W, feature_dim)

        # Project feature_dim (2048) to embed_size if they are different
        # This output will be (batch, num_pixels, embed_size)
        features = self.spatial_feature_projection(features)
        
        return features # This `features` will now be (batch, num_pixels, embed_size)


class DecoderRNN(nn.Module):
    """RNN decoder using LSTM (no attention)."""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.num_layers = num_layers
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob) # Dropout layer
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        # For DecoderRNN (no attention), features is expected to be (batch, embed_size)
        # If EncoderCNN outputs spatial features, you'd need to average pool them here
        # or modify this decoder to accept spatial features and then pool.
        # For simplicity, if using this decoder, ensure encoder provides (batch, embed_size)
        if features.dim() == 3: # If EncoderCNN is updated to output spatial features
            features = features.mean(dim=1) # Average pool spatial features to (batch, embed_size)

        embeds = self.embedding(captions[:, :-1])
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)
        x, _ = self.lstm(x)
        x = self.fcn(self.drop(x)) # Apply dropout before final linear layer
        return x

    def generate_caption(self, inputs, hidden=None, max_len=20, vocab=None):
        # If inputs are spatial features, average pool for this decoder
        if inputs.dim() == 3:
            inputs = inputs.mean(dim=1) # (batch, embed_size)
        
        batch_size = inputs.size(0)
        captions = []
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1) # [batch, 1, embed_size]

        if hidden is None:
            device = inputs.device
            h = torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size, device=device)
            c = torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size, device=device)
            hidden = (h, c)
            
        for i in range(max_len):
            output, hidden = self.lstm(inputs, hidden)  # output: [batch, 1, hidden_size]
            output = self.fcn(self.drop(output.squeeze(1))) # Apply dropout here too
            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            inputs = self.embedding(predicted_word_idx).unsqueeze(1) # [batch, 1, embed_size]
        return [vocab.itos[idx] for idx in captions]


class Attention(nn.Module):
    """Additive (Bahdanau) Attention mechanism."""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch, num_pixels, encoder_dim) - This is now the expected input from EncoderCNN
        # decoder_hidden: (batch, decoder_dim)
        att1 = self.encoder_att(encoder_out)  # (batch, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (batch, 1, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch, num_pixels)
        alpha = self.softmax(att)  # (batch, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch, encoder_dim)
        return attention_weighted_encoding, alpha

class DecoderRNNWithAttention(nn.Module):
    """LSTM Decoder with Attention."""
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=None, attention_dim=256, num_layers=1, drop_prob=0.3):
        super(DecoderRNNWithAttention, self).__init__()
        # encoder_dim should now be the output dimension of the spatial features from EncoderCNN
        # If EncoderCNN projects to embed_size, then encoder_dim = embed_size
        self.encoder_dim = encoder_dim if encoder_dim is not None else embed_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attention = Attention(self.encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTMCell is for single-step processing. For multi-layer, you'd stack them or use nn.LSTM
        # If num_layers > 1, consider using nn.LSTM and handling hidden states across layers
        self.lstm = nn.LSTMCell(embed_size + self.encoder_dim, hidden_size) 
        
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob) # Dropout layer
        self.num_layers = num_layers # LSTMCell implicitly is 1 layer, for multiple layers, you'd need to chain LSTMCells or use nn.LSTM

        # Linear layer to initialize LSTM hidden state from encoder features if needed
        self.init_h = nn.Linear(self.encoder_dim, hidden_size)
        self.init_c = nn.Linear(self.encoder_dim, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, features, captions):
        # features: (batch, num_pixels, encoder_dim) - Expected from updated EncoderCNN
        batch_size = features.size(0)
        embeddings = self.embedding(captions[:, :-1]) # (batch, seq_len, embed_size)
        
        # Initialize hidden and cell states from encoder features
        h, c = self.init_hidden_state(features.mean(dim=1)) # Use mean of spatial features for initialization

        outputs = torch.zeros(batch_size, embeddings.size(1), self.vocab_size).to(features.device)
        
        for t in range(embeddings.size(1)):
            word_embed = embeddings[:, t, :]
            context, alpha = self.attention(features, h) # Pass spatial features to attention
            lstm_input = torch.cat([word_embed, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            out = self.fcn(self.drop(h)) # Apply dropout before final linear layer
            outputs[:, t, :] = out
        return outputs

    def init_hidden_state(self, encoder_out_mean):
        # Initialize hidden/cell state with a projection of the mean encoder output
        # encoder_out_mean: (batch, encoder_dim)
        h = self.tanh(self.init_h(encoder_out_mean))
        c = self.tanh(self.init_c(encoder_out_mean))
        return h, c

    def generate_caption(self, features, max_len=20, vocab=None, return_attention=False):
        # features: (batch, num_pixels, encoder_dim) - Expected from updated EncoderCNN
        batch_size = features.size(0)
        
        # Initialize hidden and cell states
        h, c = self.init_hidden_state(features.mean(dim=1)) # Use mean of spatial features for initialization
        
        inputs = self.embedding(torch.full((batch_size,), vocab.stoi["<SOS>"], dtype=torch.long, device=features.device)) # <SOS>
        
        captions = []
        attention_weights_history = []
        
        for i in range(max_len):
            context, alpha = self.attention(features, h) # Pass spatial features to attention
            if return_attention:
                attention_weights_history.append(alpha.cpu().numpy()) # alpha: (batch, num_pixels)
            
            lstm_input = torch.cat([inputs, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            
            out = self.fcn(self.drop(h))
            predicted = out.argmax(dim=1)
            
            captions.append(predicted.item())
            
            if vocab.itos[predicted.item()] == "<EOS>":
                break
            
            inputs = self.embedding(predicted)
        
        result = [vocab.itos[idx] for idx in captions]
        if return_attention:
            # For visualization, ensure attention_weights_history is a list of (num_pixels,) arrays
            return result, [w[0] for w in attention_weights_history] # Assuming batch_size=1 for generation
        return result


class EncoderDecoder(nn.Module):
    """Complete encoder-decoder model. Supports LSTM and LSTM+Attention."""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3, decoder_type='lstm', attention_dim=256):
        super(EncoderDecoder, self).__init__()
        # EncoderCNN now outputs (batch, num_pixels, embed_size)
        self.encoder = EncoderCNN(embed_size)
        
        if decoder_type == 'attention':
            # encoder_dim for attention decoder should be 'embed_size' from EncoderCNN output
            self.decoder = DecoderRNNWithAttention(embed_size, hidden_size, vocab_size, encoder_dim=embed_size, attention_dim=attention_dim, num_layers=num_layers, drop_prob=drop_prob)
        else:
            # If using plain LSTM, it expects (batch, embed_size)
            # The DecoderRNN's forward method will now handle average pooling if spatial features are passed
            self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images) # features will be (batch, num_pixels, embed_size)
        outputs = self.decoder(features, captions)
        return outputs


def show_image(inp, title=None):
    """Display image with optional title."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def create_attention_visualization(model, dataset, device, vocab, num_samples=3, save_dir=None, dataset_type='flickr30k'):
    """Create attention visualizations for the attention-based model."""
    if not hasattr(model.decoder, 'attention'):
        print("Model does not have attention mechanism. Skipping attention visualization.")
        return
    
    model.eval()
    print(f"\nCreating attention visualizations for {num_samples} samples...")
    
    random.seed(42)
    dataset_size = len(dataset)
    random_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    for i, idx in enumerate(random_indices):
        img, _ = dataset[idx]
        img_tensor = img.unsqueeze(0).to(device) # Add batch dimension for inference
        
        with torch.no_grad():
            # Encoder now outputs spatial features (batch, num_pixels, embed_size)
            features = model.encoder(img_tensor) 
            # generate_caption now returns attention weights
            caption_tokens, attention_weights_list = model.decoder.generate_caption(
                features, vocab=vocab, return_attention=True
            )
        
        caption_text = ' '.join(caption_tokens)
        print(f"Sample {i+1} - Generated Caption: {caption_text}")

        # Assuming attention_weights_list is a list of (num_pixels,) numpy arrays
        # The num_pixels should correspond to the spatial grid of the encoder (e.g., 49 for 7x7)
        # You'll need to know the H_prime, W_prime from your EncoderCNN's output
        # For ResNet50 layer4, it's typically 7x7, so num_pixels = 49
        H_att, W_att = 7, 7 # Assuming ResNet50's last conv layer output spatial dimensions

        # Call the updated visualize_attention_heatmap function
        visualize_attention_heatmap(
            image=img, 
            caption_tokens=caption_tokens, 
            attention_weights=attention_weights_list, 
            save_path=os.path.join(save_dir, f'attention_viz_sample_{i+1}.png'),
            H_att=H_att, W_att=W_att # Pass spatial dimensions for reshaping
        )
            
    print("Attention visualization completed!")


def visualize_attention_heatmap(image, caption_tokens, attention_weights, save_path=None, H_att=7, W_att=7):
    """
    Create a detailed attention heatmap visualization.
    Args:
        image (torch.Tensor): The original image tensor (C, H, W).
        caption_tokens (list): List of generated words.
        attention_weights (list): List of numpy arrays, each (num_pixels,) for a word.
        save_path (str, optional): Path to save the plot.
        H_att (int): Height of the attention map grid (e.g., 7 for ResNet50).
        W_att (int): Width of the attention map grid (e.g., 7 for ResNet50).
    """
    if len(caption_tokens) != len(attention_weights):
        print(f"Warning: Mismatch between caption length ({len(caption_tokens)}) and attention weights ({len(attention_weights)})")
        # Adjust attention_weights to match caption length, e.g., by truncating or padding
        min_len = min(len(caption_tokens), len(attention_weights))
        caption_tokens = caption_tokens[:min_len]
        attention_weights = attention_weights[:min_len]

    n_words = len(caption_tokens)
    n_cols = 5 # Max 5 columns for readability
    n_rows = int(np.ceil(n_words / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    # Flatten axes array for easy iteration if n_rows is 1
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (word, attn) in enumerate(zip(caption_tokens, attention_weights)):
        row = i // n_cols
        col = i % n_cols
        
        # Show original image
        axes[row, col].imshow(image.permute(1, 2, 0).cpu().numpy()) # Convert to numpy for imshow
        
        # Overlay attention heatmap
        # Reshape attention to the spatial grid (e.g., 7x7)
        if attn.ndim == 1 and attn.shape[0] == H_att * W_att:
            attn_reshaped = attn.reshape(H_att, W_att)
        elif attn.ndim == 0: # Handle case where attn might be a scalar if num_pixels=1
            attn_reshaped = np.array([[attn]]) # Reshape to 1x1 for imshow
            H_att, W_att = 1, 1 # Adjust for extent
        else:
            print(f"Warning: Unexpected attention weight shape {attn.shape}. Skipping heatmap for word '{word}'.")
            axes[row, col].set_title(f'"{word}" (No Heatmap)', fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
            continue

        # Resize attention map to image dimensions
        # Use OpenCV for resizing for better quality, ensure it's installed (pip install opencv-python)
        try:
            import cv2
            attn_resized = cv2.resize(attn_reshaped, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            print("Warning: OpenCV not found. Using PIL for image resizing (might be slower/lower quality).")
            attn_resized = Image.fromarray((attn_reshaped * 255).astype(np.uint8)).resize((image.shape[2], image.shape[1]), Image.BILINEAR)
            attn_resized = np.array(attn_resized) / 255.0 # Normalize back to 0-1
        
        # Normalize to 0-1 for colormap
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        axes[row, col].imshow(attn_resized, cmap='jet', alpha=0.7, extent=[0, image.shape[2], image.shape[1], 0])
        axes[row, col].set_title(f'"{word}"', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for j in range(n_words, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved to: {save_path}")
    
    plt.show()
    plt.close()


def create_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Create a learning rate scheduler based on the specified type.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler to create
        **kwargs: Additional arguments for specific schedulers
    
    Returns:
        PyTorch scheduler or None
    """
    if scheduler_type == 'reduce_lr_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            patience=kwargs.get('patience', 3),
            factor=kwargs.get('factor', 0.5),
            min_lr=kwargs.get('min_lr', 1e-7)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 7),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('t_max', 10),
            eta_min=kwargs.get('min_lr', 1e-7)
        )
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type == 'one_cycle':
        # OneCycleLR requires total steps, so we'll need to calculate it
        # This is a placeholder - actual implementation would need total steps
        # It's best to pass steps_per_epoch from the DataLoader length
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 0.001),
            epochs=kwargs.get('epochs', 10),
            steps_per_epoch=kwargs.get('steps_per_epoch', 100), # MANDATORY: Pass len(train_loader) here
            pct_start=kwargs.get('warmup_epochs', 0) / kwargs.get('epochs', 10) if kwargs.get('epochs', 10) > 0 else 0.0 # Handle division by zero
        )
    else:
        print(f"Warning: Unknown scheduler type '{scheduler_type}'. No scheduler will be used.")
        return None


def get_current_lr(optimizer):
    """Get the current learning rate from the optimizer."""
    return optimizer.param_groups[0]['lr']


def log_lr_change(epoch, optimizer, scheduler_type, val_loss=None):
    """Log learning rate changes."""
    current_lr = get_current_lr(optimizer)
    if scheduler_type == 'reduce_lr_on_plateau':
        print(f"Epoch {epoch}: Learning Rate = {current_lr:.6f}, Validation Loss = {val_loss:.5f}")
    else:
        print(f"Epoch {epoch}: Learning Rate = {current_lr:.6f}")


def create_run_directory(base_dir, dataset_type, decoder_type, **kwargs):
    """
    Create a unique directory for each training run with timestamp and parameter summary.
    
    Args:
        base_dir: Base directory for saving models
        dataset_type: Type of dataset (flickr8k/flickr30k)
        decoder_type: Type of decoder (lstm/attention)
        **kwargs: Additional parameters to include in directory name
    
    Returns:
        Unique directory path for this run
    """
    import json
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    param_parts = []
    
    if 'epochs' in kwargs:
        param_parts.append(f"ep{kwargs['epochs']}")
    if 'batch_size' in kwargs:
        param_parts.append(f"bs{kwargs['batch_size']}")
    if 'learning_rate' in kwargs:
        lr_str = f"{kwargs['learning_rate']:.0e}".replace('e-0', 'e').replace('e+0', 'e')
        param_parts.append(f"lr{lr_str}")
    if 'use_scheduler' in kwargs and kwargs['use_scheduler']:
        scheduler_type = kwargs.get('scheduler_type', 'unknown')
        param_parts.append(f"{scheduler_type}")
    if 'save_attention_viz' in kwargs and kwargs['save_attention_viz']:
        param_parts.append("att")
    
    param_str = "_".join(param_parts) if param_parts else "default"
    run_name = f"{dataset_type}_{decoder_type}_{param_str}_{timestamp}"
    
    encoder_type = 'resnet'
    run_dir = os.path.join(base_dir, dataset_type, f"{encoder_type}_{decoder_type}", run_name)
    
    os.makedirs(run_dir, exist_ok=True)
    
    config = {
        'dataset_type': dataset_type,
        'decoder_type': decoder_type,
        'timestamp': timestamp,
        'run_name': run_name,
        **kwargs
    }
    
    config_file = os.path.join(run_dir, 'run_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    summary_file = os.path.join(run_dir, 'run_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Training Run Summary\n")
        f.write(f"===================\n\n")
        f.write(f"Run Directory: {run_name}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Decoder: {decoder_type}\n")
        
        for key, value in kwargs.items():
            if key not in ['dataset_type', 'decoder_type']:
                f.write(f"{key}: {value}\n")
        
        f.write(f"\nFull configuration saved to: run_config.json\n")
    
    print(f"Created run directory: {run_dir}")
    print(f"Configuration saved to: {config_file}")
    
    return run_dir


def create_data_loader(batch_size=4, num_workers=0, dataset_type='flickr8k'):
    """Create data loader with transforms for specified dataset."""
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_type]
    
    dataset = CustomDataset(
        root_dir=config['image_dir'],
        captions_file=config['captions_file'],
        transform=transforms,
        dataset_type=dataset_type
    )
    
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
    )
    
    return data_loader, dataset

def eval_intermediate_model_dur_train(epoch, model, data_loader, dataset, device):
    print(f"\nEpoch: {epoch} - Evaluating intermediate model...")

    model.eval()
    with torch.no_grad():
        dataiter = iter(data_loader)
        img, _ = next(dataiter)
        # Assuming generate_caption for non-attention decoder still takes (batch, embed_size)
        # So, we average pool the spatial features from the updated EncoderCNN
        features = model.encoder(img[0:1].to(device)).mean(dim=1) 
        caps = model.decoder.generate_caption(features, vocab=dataset.vocab)
        caption = ' '.join(caps)
        print(f"Sample caption: {caption}")
        show_image(img[0], title=caption)

    model.train()

def train_model(model, data_loader, dataset, device, num_epochs=20, learning_rate=0.0001, print_every=2000, dataset_type='flickr8k', val_fraction=0.2, decoder_type='lstm', start_epoch=1, optimizer=None, scheduler=None):
    """Train the model with model saving and loss tracking, and visualize loss after each epoch."""
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    vocab_size = len(dataset.vocab)
    
    encoder_type = 'resnet'
    model_save_dir = os.path.join(MODEL_SAVE_DIR, dataset_type, f"{encoder_type}_{decoder_type}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    n_total = len(dataset)
    n_val = int(val_fraction * n_total)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    pin_memory = (device.type == 'cuda' or device.type == 'mps')
    
    train_loader = DataLoader(
        train_set, 
        batch_size=data_loader.batch_size, 
        shuffle=True, 
        num_workers=data_loader.num_workers, 
        collate_fn=data_loader.collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True if data_loader.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=data_loader.batch_size, 
        shuffle=False, 
        num_workers=data_loader.num_workers, 
        collate_fn=data_loader.collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True if data_loader.num_workers > 0 else False
    )
    
    training_losses = []
    validation_losses = []
    learning_rates = []
    epochs_list = []
    metrics_history = []
    previous_epochs = start_epoch - 1
    metrics_path = os.path.join(model_save_dir, 'metrics_history.json')
    if previous_epochs > 0 and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            prev_metrics = json.load(f)
            metrics_history = prev_metrics if isinstance(prev_metrics, list) else []
        loss_curve_path = os.path.join(model_save_dir, 'loss_curve.json')
        if os.path.exists(loss_curve_path):
            with open(loss_curve_path, 'r') as f:
                prev_loss = json.load(f)
                training_losses = prev_loss.get('training_losses', [])
                validation_losses = prev_loss.get('validation_losses', [])
                learning_rates = prev_loss.get('learning_rates', [])
                epochs_list = prev_loss.get('epochs', [])
        else:
            epochs_list = list(range(1, previous_epochs + 1))
            training_losses = [None] * previous_epochs
            validation_losses = [None] * previous_epochs
            learning_rates = [None] * previous_epochs
    
    ref_save_path = os.path.join(model_save_dir, 'reference_indices.json')
    reference_indices = load_reference_indices(ref_save_path)
    if reference_indices is None:
        reference_indices = select_reference_images(dataset, n=10, save_path=ref_save_path)

    total_epochs = previous_epochs + num_epochs
    for eidx, epoch in enumerate(range(start_epoch, start_epoch + num_epochs), start=1):
        model.train()
        epoch_loss = 0.0
        progress_msg = f"Epoch {epoch}/{total_epochs}"
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory before epoch {epoch}: {gpu_memory_before:.2f} GB")
        
        epoch_start_time = time.time()
        batch_times = []
        
        for i, (images, captions) in enumerate(tqdm(train_loader, desc=progress_msg)):
            batch_start_time = time.time()
            
            images, captions = images.to(device, non_blocking=True), captions.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images, captions)
            
            # For LSTM decoder (no attention), outputs has one extra time step due to prepended image feature
            # The forward method of DecoderRNN now handles avg pooling if spatial features are passed
            # So, the reshaping logic here remains the same as outputs from decoder will be (batch, seq_len, vocab_size)
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            loss.backward()
            
            # Gradient clipping (Good practice to prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
          
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        
        tqdm.write(f"Epoch {epoch} finished. Final batch loss: {loss.item():.4f}")
        tqdm.write(f"Epoch time: {epoch_time:.2f}s, Avg batch time: {avg_batch_time:.3f}s")
        
        if device.type == 'cuda':
            gpu_memory_after = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory after epoch {epoch}: {gpu_memory_after:.2f} GB")
            print(f"GPU Memory used: {gpu_memory_after - gpu_memory_before:.2f} GB")
        
        training_losses.append(avg_epoch_loss)
        epochs_list.append(epoch)

        val_loss = calculate_validation_loss(model, val_loader, criterion, vocab_size, device, val_fraction=1.0)
        validation_losses.append(val_loss)
        print(f"Epoch {epoch} completed ({epoch}/{total_epochs}):")
        print(f"  Training Loss: {avg_epoch_loss:.5f}")
        print(f"  Validation Loss: {val_loss:.5f}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            current_lr = get_current_lr(optimizer)
            learning_rates.append(current_lr)
            print(f"  Learning Rate: {current_lr:.6f}")
            
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                print(f"  Scheduler: ReduceLROnPlateau (patience={scheduler.patience}, factor={scheduler.factor})")
            elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                print(f"  Scheduler: StepLR (step_size={scheduler.step_size}, gamma={scheduler.gamma})")
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                print(f"  Scheduler: CosineAnnealingLR (T_max={scheduler.T_max})")
            elif isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
                print(f"  Scheduler: ExponentialLR (gamma={scheduler.gamma})")
            elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                print(f"  Scheduler: OneCycleLR (max_lr={scheduler.max_lr}, pct_start={scheduler.pct_start})") # Add more details
        else:
            current_lr = get_current_lr(optimizer)
            learning_rates.append(current_lr)
            print(f"  Learning Rate: {current_lr:.6f} (no scheduler)")

        plt.figure(figsize=(8,4))
        plt.plot(epochs_list, training_losses, label='Train Loss')
        plt.plot(epochs_list, validation_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training/Validation Loss')
        plt.show()
        plt.close()

        print("Evaluating BLEU, WER, ROUGE on validation set...")
        metrics = evaluate_on_validation(model, val_loader, dataset, dataset.vocab, device, n_samples=100 if len(val_set) > 100 else None)
        metrics_history.append(metrics)
        print(f"  BLEU: {metrics['bleu']:.4f}  WER: {metrics['wer']:.4f}  ROUGE: {metrics['rouge']:.4f}")

        is_best = val_loss == min(validation_losses) if len(validation_losses) > 0 else False
        save_model_checkpoint(model, optimizer, scheduler, epoch, avg_epoch_loss, val_loss, metrics, model_save_dir, vocab_size, decoder_type, dataset, best_loss=min(validation_losses) if validation_losses else None, is_best=is_best)
        print(f"  Model checkpoint saved (best: {is_best})")

        save_loss_curves(epochs_list, training_losses, validation_losses, model_save_dir)
        with open(os.path.join(model_save_dir, 'loss_curve.json'), 'w') as f:
            json.dump({'epochs': epochs_list, 'training_losses': training_losses, 'validation_losses': validation_losses, 'learning_rates': learning_rates}, f, indent=2)
        save_metrics_history(metrics_history, model_save_dir)

        if hasattr(model, 'decoder'):
            imgs, gts, results = run_caption_comparison({decoder_type: model}, dataset, reference_indices, dataset.vocab, device, [decoder_type])
            plot_caption_comparison(imgs, gts, results, [decoder_type], output_dir=model_save_dir)

        if decoder_type == 'attention' and args.save_attention_viz:
            # Ensure the visualization functions are updated to handle spatial features
            # create_attention_visualization is now called in main for better control
            pass # This part is handled by the call in main(args)
            
    model_info = {
        'epochs': epochs_list,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'learning_rates': learning_rates,
        'metrics_history': metrics_history,
        'dataset_type': dataset_type
    }
    with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    create_loss_plots(model_info, model_save_dir)
    print("Training completed!")
    return model_info


def calculate_validation_loss(model, data_loader, criterion, vocab_size, device, val_fraction=0.1):
    """Calculate validation loss using a subset of the data."""
    model.eval()
    total_val_loss = 0.0
    val_batches = 0
    
    num_val_batches = max(1, int(len(data_loader) * val_fraction))
    
    with torch.no_grad():
        for i, (image, captions) in enumerate(data_loader):
            if i >= num_val_batches:
                break
                
            image, captions = image.to(device), captions.to(device)
            outputs = model(image, captions)
            
            # For LSTM decoder (no attention), outputs has one extra time step due to prepended image feature
            # The forward method of DecoderRNN now handles avg pooling if spatial features are passed
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            total_val_loss += loss.item()
            val_batches += 1
    
    model.train()
    return total_val_loss / val_batches if val_batches > 0 else float('inf')


def create_loss_plots(model_info, save_dir):
    """Create and save training/validation loss plots and learning rate plot."""
    epochs = model_info['epochs']
    train_losses = model_info['training_losses']
    val_losses = model_info['validation_losses']
    learning_rates = model_info.get('learning_rates', [])
    dataset_type = model_info.get('dataset_type', 'unknown')
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {dataset_type.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss Over Time - {dataset_type.upper()}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_losses, 'r-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss Over Time - {dataset_type.upper()}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    plt.plot(epochs, loss_diff, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('|Training Loss - Validation Loss|')
    plt.title(f'Loss Difference (Overfitting Indicator) - {dataset_type.upper()}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_losses.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    if learning_rates:
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Learning Rate Over Time - {dataset_type.upper()}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Learning rate plot saved to {os.path.join(save_dir, 'learning_rate.png')}")
    
    print(f"Loss plots saved to {os.path.join(save_dir, 'training_losses.png')}")


def load_trained_model(model_path, device):
    """Load a trained model from saved checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    hyperparams = checkpoint['hyperparameters']
    model = EncoderDecoder(
        embed_size=hyperparams['embed_size'],
        hidden_size=hyperparams['hidden_size'],
        vocab_size=hyperparams['vocab_size'],
        num_layers=hyperparams['num_layers'],
        decoder_type=hyperparams.get('decoder_type', 'lstm'),
        attention_dim=hyperparams.get('attention_dim', 256)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    vocab = checkpoint.get('vocab', None)
    if vocab is None:
        print("WARNING: No vocab found in checkpoint. You must provide the correct vocabulary manually!")
    return model, vocab, checkpoint


def test_model(model, dataset, device, num_samples=5, dataset_type='flickr8k'):
    """Test the trained model on randomly sampled images."""
    model.eval()
    
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    config = DATASET_CONFIGS[dataset_type]
    
    test_dataset = CustomDataset(
        root_dir=config['image_dir'],
        captions_file=config['captions_file'],
        transform=transforms,
        dataset_type=dataset_type
    )
    
    print(f"\nTesting model on {num_samples} randomly sampled images from {dataset_type}:")
    print("-" * 50)
    
    random.seed(42)
    dataset_size = len(test_dataset)
    random_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    for i, idx in enumerate(random_indices):
        img_original_tensor, _ = test_dataset[idx] # Keep original tensor for visualization
        img_input_tensor = img_original_tensor.unsqueeze(0).to(device) # Add batch dim for model input
        
        with torch.no_grad():
            features = model.encoder(img_input_tensor)
            
            if hasattr(model.decoder, 'attention'):
                # For attention decoder, generate_caption returns attention weights
                caption_tokens, attention_weights_list = model.decoder.generate_caption(
                    features, vocab=dataset.vocab, return_attention=True
                )
                caption_text = ' '.join(caption_tokens)
                print(f"Image {i+1} (index {idx}): {caption_text}")

                # Visualize attention for test samples
                H_att, W_att = 7, 7 # Assuming ResNet50's last conv layer output spatial dimensions
                visualize_attention_heatmap(
                    image=img_original_tensor, 
                    caption_tokens=caption_tokens, 
                    attention_weights=attention_weights_list, 
                    save_path=None, # Don't save during test_model by default, just show
                    H_att=H_att, W_att=W_att
                )

            else:
                # For non-attention decoder, average pool features for generate_caption
                features_pooled = features.mean(dim=1) 
                caption_tokens = model.decoder.generate_caption(features_pooled, vocab=dataset.vocab)
                caption_text = ' '.join(caption_tokens)
                print(f"Image {i+1} (index {idx}): {caption_text}")
                show_image(img_original_tensor, title=caption_text)
                plt.show()
                plt.close()


def main(args):
    """Main function to run the image captioning pipeline."""
    global MODEL_SAVE_DIR
    print("Flickr Image Captioning with PyTorch ResNet-LSTM")
    print("=" * 50)
    
    print("=== GPU Debug Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("=============================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Selected device: {device}")
    
    if device.type == 'cuda':
        print(f"Model will be trained on GPU: {torch.cuda.get_device_name()}")
    else:
        print("WARNING: Model will be trained on CPU - this will be very slow!")
    
    print(f"Using dataset: {args.dataset_type}")
    print(f"Decoder type: {args.decoder}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    
    # Set global paths from args
    global PRETRAINED_RESNET_MODEL_PATH
    PRETRAINED_RESNET_MODEL_PATH = args.resnet_weights
    MODEL_SAVE_DIR = args.model_save_dir

    # Override dataset configs if provided
    if args.image_dir is not None:
        DATASET_CONFIGS[args.dataset_type]['image_dir'] = args.image_dir
    if args.captions_file is not None:
        DATASET_CONFIGS[args.dataset_type]['captions_file'] = args.captions_file

    if args.mode == 'test' and args.model_path:
        print(f"Loading model from: {args.model_path}")
        model, vocab, checkpoint = load_trained_model(args.model_path, device)
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Training Loss: {checkpoint['training_loss']:.5f}")
        print(f"Validation Loss: {checkpoint['validation_loss']:.5f}")
        
        # In test mode, create a dataset instance to pass to test_model
        # The vocab will be loaded from the checkpoint
        test_data_loader, test_dataset = create_data_loader(batch_size=args.batch_size, num_workers=args.num_workers, dataset_type=args.dataset_type)
        test_dataset.vocab = vocab # Ensure the vocab from loaded model is used
        test_model(model, test_dataset, device, num_samples=args.test_samples, dataset_type=args.dataset_type) # Use args.test_samples
    else:
        print("Creating data loader...")
        data_loader, dataset = create_data_loader(batch_size=args.batch_size, num_workers=args.num_workers, dataset_type=args.dataset_type)
        print(f"Dataset size: {len(dataset)}")
        print(f"Vocabulary size: {len(dataset.vocab)}")
        
        embed_size = args.embed_size
        hidden_size = args.hidden_size
        vocab_size = len(dataset.vocab)
        num_layers = args.num_layers
        attention_dim = args.attention_dim
        drop_prob = args.drop_prob
        
        start_epoch = 1
        optimizer_instance = None # Initialize as None
        scheduler_instance = None # Initialize as None

        if args.model_path:
            print(f"Resuming training from: {args.model_path}")
            model, vocab, checkpoint = load_trained_model(args.model_path, device)
            dataset.vocab = vocab
            
            optimizer_instance = torch.optim.Adam(model.parameters()) # Re-initialize optimizer
            if 'optimizer_state_dict' in checkpoint:
                optimizer_instance.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                # Recreate scheduler with correct parameters and load state
                scheduler_kwargs = {
                    'patience': args.scheduler_patience,
                    'factor': args.scheduler_factor,
                    'step_size': args.scheduler_step_size,
                    'gamma': args.scheduler_gamma,
                    't_max': args.scheduler_t_max,
                    'min_lr': args.min_lr,
                    'epochs': args.epochs,
                    'steps_per_epoch': len(data_loader), # Crucial for OneCycleLR
                    'warmup_epochs': args.warmup_epochs
                }
                scheduler_instance = create_scheduler(optimizer_instance, args.scheduler_type, **scheduler_kwargs)
                scheduler_instance.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Resumed scheduler: {scheduler_instance}")
            else:
                print("No scheduler state found in checkpoint or scheduler not used.")
            start_epoch = checkpoint['epoch'] + 1
        else:
            print(f"Initializing model with embed_size={embed_size}, hidden_size={hidden_size}, vocab_size={vocab_size}, num_layers={num_layers}, decoder_type={args.decoder}, attention_dim={attention_dim}, drop_prob={drop_prob}")
            model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers, decoder_type=args.decoder, attention_dim=attention_dim, drop_prob=drop_prob).to(device)
            optimizer_instance = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            
            if args.use_scheduler:
                print(f"Creating {args.scheduler_type} scheduler...")
                scheduler_kwargs = {
                    'patience': args.scheduler_patience,
                    'factor': args.scheduler_factor,
                    'step_size': args.scheduler_step_size,
                    'gamma': args.scheduler_gamma,
                    't_max': args.scheduler_t_max,
                    'min_lr': args.min_lr,
                    'epochs': args.epochs,
                    'steps_per_epoch': len(data_loader), # MANDATORY for OneCycleLR
                    'warmup_epochs': args.warmup_epochs
                }
                scheduler_instance = create_scheduler(optimizer_instance, args.scheduler_type, **scheduler_kwargs)
                print(f"Scheduler created: {scheduler_instance}")
            else:
                print("No scheduler will be used (constant learning rate)")
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Initial learning rate: {get_current_lr(optimizer_instance):.6f}")
        print("\nStarting training...")
        
        run_params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'attention_dim': attention_dim,
            'drop_prob': drop_prob,
            'num_workers': args.num_workers,
            'use_scheduler': args.use_scheduler,
            'scheduler_type': args.scheduler_type if args.use_scheduler else None,
            'scheduler_patience': args.scheduler_patience,
            'scheduler_factor': args.scheduler_factor,
            'scheduler_step_size': args.scheduler_step_size,
            'scheduler_gamma': args.scheduler_gamma,
            'scheduler_t_max': args.scheduler_t_max,
            'min_lr': args.min_lr,
            'warmup_epochs': args.warmup_epochs,
            'save_attention_viz': args.save_attention_viz,
            'attention_viz_samples': args.attention_viz_samples,
            'advanced_attention_viz': args.advanced_attention_viz,
            'test_samples': args.test_samples # Include test_samples in run_params
        }
        
        run_dir = create_run_directory(MODEL_SAVE_DIR, args.dataset_type, args.decoder, **run_params)
        
        original_model_save_dir = MODEL_SAVE_DIR
        MODEL_SAVE_DIR = run_dir
        
        training_info = train_model(
            model, data_loader, dataset, device, 
            num_epochs=args.epochs, 
            print_every=1000, 
            learning_rate=args.learning_rate,
            dataset_type=args.dataset_type, 
            start_epoch=start_epoch, 
            optimizer=optimizer_instance, 
            scheduler=scheduler_instance, 
            decoder_type=args.decoder
        )
        
        MODEL_SAVE_DIR = original_model_save_dir
        
        print("\nTesting model...")
        test_model(model, dataset, device, num_samples=args.test_samples, dataset_type=args.dataset_type)
        
        if args.decoder == 'attention' and args.save_attention_viz:
            print("\nCreating attention visualizations...")
            create_attention_visualization(
                model, dataset, device, dataset.vocab, 
                num_samples=args.attention_viz_samples, 
                save_dir=run_dir,
                dataset_type=args.dataset_type
            )
            
            if args.advanced_attention_viz:
                print("\nCreating advanced attention visualizations...")
                # The create_advanced_attention_visualization function is now redundant
                # as visualize_attention_heatmap handles the advanced plotting.
                # You can remove create_advanced_attention_visualization if you only use visualize_attention_heatmap
                # For now, I'll keep the call but it will use the same core visualization logic
                # as create_attention_visualization now calls visualize_attention_heatmap.
                # If you need a *different* advanced visualization, you'd implement it separately.
                # For consistency, ensure this also calls visualize_attention_heatmap with appropriate parameters.
                # For simplicity, I'll just call create_attention_visualization again if advanced_attention_viz is true
                # for now, but ideally, the logic would be merged or distinct.
                # As a placeholder, let's just ensure the main viz is called.
                pass # Already handled by create_attention_visualization if save_attention_viz is true

    print("\nPipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flickr Image Captioning with PyTorch ResNet-LSTM")
    parser.add_argument('--dataset_type', type=str, default=DEFAULT_DATASET, choices=['flickr8k','flickr30k'], help='Dataset to use (flickr8k or flickr30k)')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory containing images')
    parser.add_argument('--captions_file', type=str, default=None, help='Captions file path')
    parser.add_argument('--model_save_dir', type=str, default="./saved_models", help='Directory to save trained models and results')
    parser.add_argument('--resnet_weights', type=str, default='../../resources/models/resnet50-19c8e357.pth', help='Path to pretrained ResNet50 weights')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--model_path', type=str, default=None, help='Path to save/load the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--decoder', type=str, default='lstm', choices=['lstm', 'attention'], help='Decoder type: lstm or attention')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading (default: 16)')
    parser.add_argument('--embed_size', type=int, default=512, help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=1024, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--attention_dim', type=int, default=512,
                        help='Attention dimension (if using attention decoder)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--drop_prob', type=float, default=0.3, help='Dropout probability')
    
    # Learning rate scheduling arguments
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, default='reduce_lr_on_plateau', 
                       choices=['reduce_lr_on_plateau', 'step', 'cosine', 'exponential', 'one_cycle'], 
                       help='Type of learning rate scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=3, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor for reducing learning rate')
    parser.add_argument('--scheduler_step_size', type=int, default=7, help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
    parser.add_argument('--scheduler_t_max', type=int, default=10, help='T_max for CosineAnnealingLR scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs for OneCycleLR')
    
    parser.add_argument('--save_attention_viz', action='store_true', help='Save attention visualizations (only for attention decoder)')
    parser.add_argument('--attention_viz_samples', type=int, default=3, help='Number of attention visualization samples')
    parser.add_argument('--advanced_attention_viz', action='store_true', help='Create advanced attention visualizations with heatmaps')
    parser.add_argument('--test_samples', type=int, default=5, help='Number of samples to test in test mode') # New argument for test samples
    
    args = parser.parse_args()

    # Set paths for backward compatibility (already handled by global vars now)
    # PRETRAINED_RESNET_MODEL_PATH = args.resnet_weights
    # MODEL_SAVE_DIR = args.model_save_dir

    # Override dataset configs if provided (already handled in main)
    # if args.image_dir is not None:
    #     DATASET_CONFIGS[args.dataset_type]['image_dir'] = args.image_dir
    # if args.captions_file is not None:
    #     DATASET_CONFIGS[args.dataset_type]['captions_file'] = args.captions_file

    main(args)

