#!/usr/bin/env python
# coding: utf-8

"""
Flickr Image Captioning with PyTorch ResNet-LSTM
================================================

This script implements an image captioning model using:
- ResNet50 as the image encoder
- LSTM as the text decoder
- Flickr8k dataset for training

Author: Image Captioning Project
"""

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
# Additional imports for metrics and visualization
from metrics_utils import compute_metrics
from attention_visualization import visualize_attention
from experiment_utils import (
    select_reference_images, load_reference_indices, save_reference_indices,
    run_caption_comparison, plot_caption_comparison, evaluate_on_validation
)
from training_utils import save_model_checkpoint, save_loss_curves, save_metrics_history

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

# Default dataset (can be overridden via command line)
DEFAULT_DATASET = 'flickr30k'

# Paths will be set via argparse
PRETRAINED_RESNET_MODEL_PATH = None
MODEL_SAVE_DIR = None

# Load spaCy model
try:
    spacy_eng = spacy.load('en_core_web_sm')
    # text = "This is a good place to find a city"
    # print([token.text.lower() for token in spacy_eng.tokenizer(text)])
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
        
        # Load captions based on dataset type
        if dataset_type == 'flickr8k':
            # Flickr8k uses txt format with image,caption columns
            self.df = pd.read_csv(captions_file, sep=',')
        elif dataset_type == 'flickr30k':
            # Flickr30k uses csv format with image,caption_number,caption,id columns
            self.df = pd.read_csv(captions_file)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        # Build vocabulary
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
        
        # Create caption vector with SOS and EOS tokens
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
    """CNN encoder using ResNet50."""
    
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=None)
        
        # Load pretrained weights
        if os.path.exists(PRETRAINED_RESNET_MODEL_PATH):
            resnet.load_state_dict(torch.load(PRETRAINED_RESNET_MODEL_PATH, map_location='cpu', weights_only=False))
        else:
            print(f"Warning: Pretrained model not found at {PRETRAINED_RESNET_MODEL_PATH}")
            print("Using randomly initialized ResNet50")
        
        # Freeze parameters
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    """RNN decoder using LSTM (no attention)."""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.num_layers = num_layers
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob)
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        embeds = self.embedding(captions[:, :-1])
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)
        x, _ = self.lstm(x)
        x = self.fcn(x)
        return x

    def generate_caption(self, inputs, hidden=None, max_len=20, vocab=None):
        batch_size = inputs.size(0)
        captions = []
        # Ensure inputs is [batch, 1, embed_size]
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        # If hidden is None, initialize to zeros
        if hidden is None:
            device = inputs.device
            h = torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size, device=device)
            c = torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size, device=device)
            hidden = (h, c)
        for i in range(max_len):
            output, hidden = self.lstm(inputs, hidden)  # output: [batch, 1, hidden_size]
            output = self.fcn(output.squeeze(1))        # [batch, vocab_size]
            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            inputs = self.embedding(predicted_word_idx).unsqueeze(1)  # [batch, 1, embed_size]
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
        # encoder_out: (batch, num_pixels, encoder_dim)
        # decoder_hidden: (batch, decoder_dim)
        att1 = self.encoder_att(encoder_out)  # (batch, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (batch, 1, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch, num_pixels)
        alpha = self.softmax(att)  # (batch, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch, encoder_dim)
        return attention_weighted_encoding, alpha

class DecoderRNNWithAttention(nn.Module):
    """LSTM Decoder with Attention."""
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=400, attention_dim=256, num_layers=1, drop_prob=0.3):
        super(DecoderRNNWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob)
        self.num_layers = num_layers

    def forward(self, features, captions):
        # features: (batch, encoder_dim) -> expand to (batch, 1, encoder_dim)
        batch_size = features.size(0)
        num_pixels = 1  # single vector from encoder
        encoder_out = features.unsqueeze(1)  # (batch, 1, encoder_dim)
        embeddings = self.embedding(captions[:, :-1])  # (batch, seq_len, embed_size)
        h, c = self.init_hidden_state(features)
        outputs = torch.zeros(batch_size, embeddings.size(1) + 1, self.vocab_size).to(features.device)
        for t in range(embeddings.size(1) + 1):
            if t == 0:
                word_embed = self.embedding(torch.full((batch_size,), 1, dtype=torch.long, device=features.device))  # <SOS>
            else:
                word_embed = embeddings[:, t-1, :]
            context, alpha = self.attention(encoder_out, h)
            lstm_input = torch.cat([word_embed, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            out = self.fcn(self.drop(h))
            outputs[:, t, :] = out
        return outputs[:, 1:, :]  # skip first (<SOS>)

    def init_hidden_state(self, encoder_out):
        # Initialize hidden/cell state with encoder output mean
        mean_encoder = encoder_out.mean(dim=1)
        h = torch.tanh(mean_encoder.new_zeros((encoder_out.size(0), self.hidden_size)))
        c = torch.tanh(mean_encoder.new_zeros((encoder_out.size(0), self.hidden_size)))
        return h, c

    def generate_caption(self, features, max_len=20, vocab=None):
        batch_size = features.size(0)
        encoder_out = features.unsqueeze(1)  # (batch, 1, encoder_dim)
        h, c = self.init_hidden_state(features)
        inputs = self.embedding(torch.full((batch_size,), 1, dtype=torch.long, device=features.device))  # <SOS>
        captions = []
        for _ in range(max_len):
            context, alpha = self.attention(encoder_out, h)
            lstm_input = torch.cat([inputs, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            out = self.fcn(self.drop(h))
            predicted = out.argmax(dim=1)
            captions.append(predicted.item())
            if vocab.itos[predicted.item()] == "<EOS>":
                break
            inputs = self.embedding(predicted)
        return [vocab.itos[idx] for idx in captions]


class EncoderDecoder(nn.Module):
    """Complete encoder-decoder model. Supports LSTM and LSTM+Attention."""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3, decoder_type='lstm', attention_dim=256):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        if decoder_type == 'attention':
            self.decoder = DecoderRNNWithAttention(embed_size, hidden_size, vocab_size, encoder_dim=embed_size, attention_dim=attention_dim, num_layers=num_layers, drop_prob=drop_prob)
        else:
            self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


def show_image(inp, title=None):
    """Display image with optional title."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def create_data_loader(batch_size=4, num_workers=0, dataset_type='flickr8k'):
    """Create data loader with transforms for specified dataset."""
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    # Get dataset configuration
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

    # Generate sample caption
    model.eval()
    with torch.no_grad():
        dataiter = iter(data_loader)
        img, _ = next(dataiter)
        features = model.encoder(img[0:1].to(device))
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
    
    # Create model save directory with dataset type
    encoder_type = 'resnet'  # currently always resnet50
    model_save_dir = os.path.join(MODEL_SAVE_DIR, dataset_type, f"{encoder_type}_{decoder_type}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Split dataset into train/val
    n_total = len(dataset)
    n_val = int(val_fraction * n_total)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=data_loader.batch_size, shuffle=True, num_workers=0, collate_fn=data_loader.collate_fn)
    val_loader = DataLoader(val_set, batch_size=data_loader.batch_size, shuffle=False, num_workers=0, collate_fn=data_loader.collate_fn)
    
    # Initialize or load loss tracking
    import json
    training_losses = []
    validation_losses = []
    epochs_list = []
    metrics_history = []
    previous_epochs = start_epoch - 1
    metrics_path = os.path.join(model_save_dir, 'metrics_history.json')
    if previous_epochs > 0 and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            prev_metrics = json.load(f)
            metrics_history = prev_metrics if isinstance(prev_metrics, list) else []
        # Try to load previous losses and epochs from loss_curve.png or another file if available
        loss_curve_path = os.path.join(model_save_dir, 'loss_curve.json')
        if os.path.exists(loss_curve_path):
            with open(loss_curve_path, 'r') as f:
                prev_loss = json.load(f)
                training_losses = prev_loss.get('training_losses', [])
                validation_losses = prev_loss.get('validation_losses', [])
                epochs_list = prev_loss.get('epochs', [])
        else:
            # Fallback: use metrics_history length
            epochs_list = list(range(1, previous_epochs + 1))
            training_losses = [None] * previous_epochs
            validation_losses = [None] * previous_epochs
    
    # Reference images for comparison
    ref_save_path = os.path.join(model_save_dir, 'reference_indices.json')
    reference_indices = load_reference_indices(ref_save_path)
    if reference_indices is None:
        reference_indices = select_reference_images(dataset, n=10, save_path=ref_save_path)

    total_epochs = previous_epochs + num_epochs
    for eidx, epoch in enumerate(range(start_epoch, start_epoch + num_epochs), start=1):
        model.train()
        epoch_loss = 0.0
        progress_msg = f"Epoch {epoch}/{total_epochs}"
        for i, (images, captions) in enumerate(tqdm(train_loader, desc=progress_msg)):
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()
            outputs = model(images, captions)
            # For LSTM decoder, outputs has one extra time step due to prepended image feature
            if hasattr(model.decoder, 'lstm') and not hasattr(model.decoder, 'attention'):
                outputs = outputs[:, 1:, :]
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch} finished. Final batch loss: {loss.item():.4f}")
        training_losses.append(avg_epoch_loss)
        epochs_list.append(epoch)

        # Validation loss
        val_loss = calculate_validation_loss(model, val_loader, criterion, vocab_size, device, val_fraction=1.0)
        validation_losses.append(val_loss)
        print(f"Epoch {epoch} completed ({epoch}/{total_epochs}):")
        print(f"  Training Loss: {avg_epoch_loss:.5f}")
        print(f"  Validation Loss: {val_loss:.5f}")

        # Plot loss after each epoch
        plt.figure(figsize=(8,4))
        plt.plot(epochs_list, training_losses, label='Train Loss')
        plt.plot(epochs_list, validation_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training/Validation Loss')
        plt.show()
        plt.close()

        # Evaluate metrics on val set
        print("Evaluating BLEU, WER, ROUGE on validation set...")
        metrics = evaluate_on_validation(model, val_loader, dataset, dataset.vocab, device, n_samples=100 if len(val_set) > 100 else None)
        metrics_history.append(metrics)
        print(f"  BLEU: {metrics['bleu']:.4f}  WER: {metrics['wer']:.4f}  ROUGE: {metrics['rouge']:.4f}")

        # Save model and best model
        is_best = val_loss == min(validation_losses) if len(validation_losses) > 0 else False
        save_model_checkpoint(model, optimizer, scheduler, epoch, avg_epoch_loss, val_loss, metrics, model_save_dir, vocab_size, decoder_type, dataset, best_loss=min(validation_losses) if validation_losses else None, is_best=is_best)
        print(f"  Model checkpoint saved (best: {is_best})")

        # Save loss curves and metrics history after each epoch
        # Save loss curves as both PNG and JSON for easier resuming
        save_loss_curves(epochs_list, training_losses, validation_losses, model_save_dir)
        # Save to JSON for future resuming
        with open(os.path.join(model_save_dir, 'loss_curve.json'), 'w') as f:
            json.dump({'epochs': epochs_list, 'training_losses': training_losses, 'validation_losses': validation_losses}, f, indent=2)
        save_metrics_history(metrics_history, model_save_dir)

        # Reference image comparison (for both decoders if available)
        if hasattr(model, 'decoder'):
            imgs, gts, results = run_caption_comparison({decoder_type: model}, dataset, reference_indices, dataset.vocab, device, [decoder_type])
            plot_caption_comparison(imgs, gts, results, [decoder_type], output_dir=model_save_dir)

        # Attention visualization for attention decoder
        if decoder_type == 'attention':
            # Only visualize attention for first reference image
            img, _ = dataset[reference_indices[0]]
            with torch.no_grad():
                features = model.encoder(img.unsqueeze(0).to(device))
                # You may need to modify generate_caption to return attention weights
                # For now, just show normal caption
                caption = model.decoder.generate_caption(features, vocab=dataset.vocab)
            print(f"Attention visualization for: {' '.join(caption)}")
            # visualize_attention(img, caption, attention_weights, dataset.vocab)  # Implement attention weight extraction

    # Save loss/metrics info
    model_info = {
        'epochs': epochs_list,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'metrics_history': metrics_history,
        'dataset_type': dataset_type
    }
    create_loss_plots(model_info, model_save_dir)
    print("Training completed!")
    return model_info
    validation_losses = []
    model_info = {
        'epochs': [],
        'training_losses': [],
        'validation_losses': [],
        'model_files': [],
        'timestamps': [],
        'dataset_type': dataset_type,
        'hyperparameters': {
            'embed_size': model.encoder.embed.out_features,
            'hidden_size': model.decoder.lstm.hidden_size,
            'vocab_size': vocab_size,
            'num_layers': model.decoder.lstm.num_layers,
            'learning_rate': learning_rate,
            'batch_size': data_loader.batch_size,
            'dataset_type': dataset_type
        }
    }
    
    print(f"Starting training for {num_epochs} epochs on {dataset_type} dataset...")
    total_batches = len(data_loader)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Create progress bar for current epoch
        pbar = tqdm(data_loader, desc=f'Epoch {epoch}/{num_epochs}', 
                   unit='batch', leave=True)
        
        for idx, (image, captions) in enumerate(pbar):
            image, captions = image.to(device), captions.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(image, captions)
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'Loss': f'{loss.item():.5f}',
                'Avg Loss': f'{epoch_loss/(idx+1):.5f}'
            })
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / total_batches
        training_losses.append(avg_epoch_loss)
        
        # Calculate validation loss (using a subset of training data for simplicity)
        val_loss = calculate_validation_loss(model, data_loader, criterion, vocab_size, device)
        validation_losses.append(val_loss)
        
        # Save model after each epoch
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_epoch_{epoch:02d}_loss_{avg_epoch_loss:.4f}_{timestamp}.pt"
        model_path = os.path.join(model_save_dir, model_filename)
        
        # Save model state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': avg_epoch_loss,
            'validation_loss': val_loss,
            'vocab': dataset.vocab,
            'hyperparameters': model_info['hyperparameters']
        }, model_path)
        
        # Update model info
        model_info['epochs'].append(epoch)
        model_info['training_losses'].append(avg_epoch_loss)
        model_info['validation_losses'].append(val_loss)
        model_info['model_files'].append(model_filename)
        model_info['timestamps'].append(timestamp)
        
        # Save training info to JSON
        with open(os.path.join(model_save_dir, 'training_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Evaluate intermediate model
        eval_intermediate_model_dur_train(epoch, model, data_loader, dataset, device)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} completed:")
        print(f"  Training Loss: {avg_epoch_loss:.5f}")
        print(f"  Validation Loss: {val_loss:.5f}")
        print(f"  Model saved: {model_filename}")
    
    # Create and save loss plots
    create_loss_plots(model_info, model_save_dir)
    
    print("Training completed!")
    return model_info


def calculate_validation_loss(model, data_loader, criterion, vocab_size, device, val_fraction=0.1):
    """Calculate validation loss using a subset of the data."""
    model.eval()
    total_val_loss = 0.0
    val_batches = 0
    
    # Use a subset of batches for validation
    num_val_batches = max(1, int(len(data_loader) * val_fraction))
    
    with torch.no_grad():
        for i, (image, captions) in enumerate(data_loader):
            if i >= num_val_batches:
                break
                
            image, captions = image.to(device), captions.to(device)
            outputs = model(image, captions)
            # For LSTM decoder, outputs has one extra time step due to prepended image feature
            if hasattr(model.decoder, 'lstm') and not hasattr(model.decoder, 'attention'):
                outputs = outputs[:, 1:, :]
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            total_val_loss += loss.item()
            val_batches += 1
    
    model.train()
    return total_val_loss / val_batches if val_batches > 0 else float('inf')


def create_loss_plots(model_info, save_dir):
    """Create and save training/validation loss plots."""
    epochs = model_info['epochs']
    train_losses = model_info['training_losses']
    val_losses = model_info['validation_losses']
    dataset_type = model_info.get('dataset_type', 'unknown')
    
    plt.figure(figsize=(12, 8))
    
    # Training and validation loss plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {dataset_type.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training loss only
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss Over Time - {dataset_type.upper()}')
    plt.grid(True, alpha=0.3)
    
    # Validation loss only
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_losses, 'r-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss Over Time - {dataset_type.upper()}')
    plt.grid(True, alpha=0.3)
    
    # Loss difference
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
    
    print(f"Loss plots saved to {os.path.join(save_dir, 'training_losses.png')}")


def load_trained_model(model_path, device):
    """Load a trained model from saved checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Recreate model with same hyperparameters
    hyperparams = checkpoint['hyperparameters']
    model = EncoderDecoder(
        embed_size=hyperparams['embed_size'],
        hidden_size=hyperparams['hidden_size'],
        vocab_size=hyperparams['vocab_size'],
        num_layers=hyperparams['num_layers'],
        decoder_type=hyperparams.get('decoder_type', 'lstm'),
        attention_dim=hyperparams.get('attention_dim', 256)
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    vocab = checkpoint.get('vocab', None)
    if vocab is None:
        print("WARNING: No vocab found in checkpoint. You must provide the correct vocabulary manually!")
    return model, vocab, checkpoint


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


def main():
    """Main function to run the image captioning pipeline."""
    print("Flickr Image Captioning with PyTorch ResNet-LSTM/LSTM-Attention")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Check if user wants to load existing model or train new one

    parser = argparse.ArgumentParser(description='Image Captioning Training/Testing')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode: train new model or test existing model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model for testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--dataset', choices=['flickr8k', 'flickr30k'], default=DEFAULT_DATASET, help='Dataset to use for training/testing')
    parser.add_argument('--decoder', choices=['lstm', 'attention'], default='lstm', help='Decoder type: lstm or attention (lstm+attention)')
    args = parser.parse_args()
    print(f"Using dataset: {args.dataset}")
    print(f"Decoder type: {args.decoder}")
    if args.mode == 'test' and args.model_path:
        print(f"Loading model from: {args.model_path}")
        model, vocab, checkpoint = load_trained_model(args.model_path, device)
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Training Loss: {checkpoint['training_loss']:.5f}")
        print(f"Validation Loss: {checkpoint['validation_loss']:.5f}")
        data_loader, dataset = create_data_loader(batch_size=32, num_workers=4, dataset_type=args.dataset)
        dataset.vocab = vocab
        test_model(model, dataset, device, num_samples=5, dataset_type=args.dataset)
    else:
        print("Creating data loader...")
        data_loader, dataset = create_data_loader(batch_size=32, num_workers=4, dataset_type=args.dataset)
        print(f"Dataset size: {len(dataset)}")
        print(f"Vocabulary size: {len(dataset.vocab)}")
        embed_size = 400
        hidden_size = 512
        vocab_size = len(dataset.vocab)
        num_layers = 2
        start_epoch = 1
        optimizer = torch.optim.Adam
        scheduler = None
        if args.model_path:
            print(f"Resuming training from: {args.model_path}")
            model, vocab, checkpoint = load_trained_model(args.model_path, device)
            dataset.vocab = vocab
            # Restore optimizer and scheduler if present
            optimizer = torch.optim.Adam(model.parameters())
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                scheduler = None
            start_epoch = checkpoint['epoch'] + 1
        else:
            print("Initializing model...")
            model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers, decoder_type=args.decoder).to(device)
            optimizer = torch.optim.Adam(model.parameters())
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("\nStarting training...")
        base_lr = 0.0001
        scaled_lr = base_lr * (32 / 16)
        training_info = train_model(model, data_loader, dataset, device, num_epochs=args.epochs, print_every=1000, learning_rate=scaled_lr, dataset_type=args.dataset, start_epoch=start_epoch, optimizer=optimizer, scheduler=scheduler, decoder_type=args.decoder)
        print("\nTesting model...")
        test_model(model, dataset, device, num_samples=3, dataset_type=args.dataset)

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
    args = parser.parse_args()

    # Set paths for backward compatibility
    PRETRAINED_RESNET_MODEL_PATH = args.resnet_weights
    MODEL_SAVE_DIR = args.model_save_dir

    # Override dataset configs if provided
    if args.image_dir is not None:
        DATASET_CONFIGS[args.dataset_type]['image_dir'] = args.image_dir
    if args.captions_file is not None:
        DATASET_CONFIGS[args.dataset_type]['captions_file'] = args.captions_file

    # Pass new args to main if needed
    main_args = {
        'mode': args.mode,
        'model_path': args.model_path,
        'epochs': args.epochs,
        'decoder': args.decoder
    }
    main(**main_args) if main.__code__.co_argcount > 0 else main()

