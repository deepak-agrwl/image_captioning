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
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import spacy
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Global variables
IMAGE_DATA_LOCATION = "../../resources/input/flickr8k/Images"
CAPTION_DATA_LOCATION = "../../resources/input/flickr8k/captions.txt"
PRETRAINED_RESNET_MODEL_PATH = '../../resources/models/resnet50-19c8e357.pth'

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
    """Custom dataset for Flickr8k images and captions."""
    
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
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
        resnet = models.resnet50(pretrained=False)
        
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
    """RNN decoder using LSTM."""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        # Vectorize the caption
        embeds = self.embedding(captions[:, :-1])
        
        # Concatenate features and captions
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)
        x, _ = self.lstm(x)
        x = self.fcn(x)
        return x

    def generate_caption(self, inputs, hidden=None, max_len=20, vocab=None):
        """Generate caption during inference."""
        batch_size = inputs.size(0)
        captions = []
        
        for i in range(max_len):
            output, hidden = self.lstm(inputs, hidden)
            output = self.fcn(output)
            output = output.view(batch_size, -1)
            
            # Select the word with highest probability
            predicted_word_idx = output.argmax(dim=1)
            
            # Save the generated word
            captions.append(predicted_word_idx.item())
            
            # End if <EOS> detected
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            
            # Send generated word as the next caption
            inputs = self.embedding(predicted_word_idx.unsqueeze(0))
        
        # Convert vocab indices to words and return sentence
        return [vocab.itos[idx] for idx in captions]


class EncoderDecoder(nn.Module):
    """Complete encoder-decoder model."""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderCNN(embed_size)
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


def create_data_loader(batch_size=4, num_workers=0):
    """Create data loader with transforms."""
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    dataset = CustomDataset(
        root_dir=IMAGE_DATA_LOCATION,
        captions_file=CAPTION_DATA_LOCATION,
        transform=transforms
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
        caps = model.decoder.generate_caption(features.unsqueeze(0), vocab=dataset.vocab)
        caption = ' '.join(caps)
        print(f"Sample caption: {caption}")
        show_image(img[0], title=caption)

    model.train()

def train_model(model, data_loader, dataset, device, num_epochs=20, learning_rate=0.0001, print_every=2000):
    """Train the model."""
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    vocab_size = len(dataset.vocab)
    
    print(f"Starting training for {num_epochs} epochs...")
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
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
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
            
            # if (idx + 1) % print_every == 0:
            #     eval_intermediate_model_dur_train(epoch, model, data_loader, dataset, device)

        eval_intermediate_model_dur_train(epoch, model, data_loader, dataset, device)
        # Print epoch summary
        avg_epoch_loss = epoch_loss / total_batches
        print(f"\nEpoch {epoch} completed - Average Loss: {avg_epoch_loss:.5f}")
    
    print("Training completed!")


def test_model(model, dataset, device, num_samples=5):
    """Test the trained model on randomly sampled images."""
    model.eval()
    
    # Create a simple data loader for testing
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    test_dataset = CustomDataset(
        root_dir=IMAGE_DATA_LOCATION,
        captions_file=CAPTION_DATA_LOCATION,
        transform=transforms
    )
    
    print(f"\nTesting model on {num_samples} randomly sampled images:")
    print("-" * 50)
    
    # Generate random indices for testing
    import random
    random.seed(42)  # For reproducible results
    dataset_size = len(test_dataset)
    random_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    for i, idx in enumerate(random_indices):
        img, _ = test_dataset[idx]
        img = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.encoder(img)
            caps = model.decoder.generate_caption(features.unsqueeze(0), vocab=dataset.vocab)
            caption = ' '.join(caps)
            
            print(f"Image {i+1} (index {idx}): {caption}")
            show_image(test_dataset[idx][0], title=caption)
            plt.show()


def main():
    """Main function to run the image captioning pipeline."""
    print("Flickr Image Captioning with PyTorch ResNet-LSTM")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Create data loader
    print("Creating data loader...")
    data_loader, dataset = create_data_loader(batch_size=32, num_workers=4)
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {len(dataset.vocab)}")
    
    # Model hyperparameters
    embed_size = 400
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    num_layers = 2
    
    # Initialize model
    print("Initializing model...")
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting training...")
    # Scale learning rate with batch size (linear scaling rule)
    base_lr = 0.0001
    scaled_lr = base_lr * (32 / 16)  # Scale from batch_size=16 to batch_size=32
    train_model(model, data_loader, dataset, device, num_epochs=12, print_every=1000, learning_rate=scaled_lr)
    
    # Test model
    print("\nTesting model...")
    test_model(model, dataset, device, num_samples=3)
    
    print("\nPipeline completed!")


if __name__ == "__main__":
    main()

