import os
import json
import random
import torch
import numpy as np
from metrics_utils import compute_metrics
from attention_visualization import visualize_attention
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

# Global variables for sentence embedding model
_tokenizer = None
_model = None
_device = None

def initialize_embedding_model(model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
    """Initialize sentence embedding model for cosine similarity computation."""
    global _tokenizer, _model, _device
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
        _device = device
        _model.to(_device)
        _model.eval()
        print(f"Initialized embedding model: {model_name} on {device}")
    except Exception as e:
        print(f"Warning: Could not load sentence transformer model: {e}")
        print("Falling back to simple word-based cosine similarity")
        _tokenizer = None
        _model = None

def get_simple_word_embedding(text, embedding_dim=300):
    """Simple word-based embedding fallback using random vectors."""
    words = text.lower().split()
    if not words:
        return np.zeros(embedding_dim)
    
    # Create simple hash-based embeddings for words
    embeddings = []
    for word in words:
        # Use hash to create consistent random vectors for each word
        np.random.seed(hash(word) % (2**32))
        word_emb = np.random.randn(embedding_dim)
        embeddings.append(word_emb)
    
    # Average word embeddings
    sentence_emb = np.mean(embeddings, axis=0)
    # Normalize
    norm = np.linalg.norm(sentence_emb)
    if norm > 0:
        sentence_emb = sentence_emb / norm
    
    return sentence_emb

def get_sentence_embedding(text):
    """Get sentence embedding using transformer model."""
    global _tokenizer, _model, _device
    
    if _tokenizer is None or _model is None:
        # Fallback to simple word-based embedding
        return get_simple_word_embedding(text)
    
    try:
        # Tokenize and encode
        inputs = _tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = _model(**inputs)
            # Use mean pooling of last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().flatten()
    except Exception as e:
        print(f"Warning: Error getting sentence embedding: {e}")
        return get_simple_word_embedding(text)

def compute_cosine_similarity(text1, text2):
    """Compute cosine similarity between two texts."""
    emb1 = get_sentence_embedding(text1)
    emb2 = get_sentence_embedding(text2)
    
    # Reshape for sklearn cosine_similarity
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)
    
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

def compute_cosine_similarities_batch(references_list, hypotheses_list):
    """Compute cosine similarities for a batch of reference-hypothesis pairs."""
    similarities = []
    
    for references, hypothesis in zip(references_list, hypotheses_list):
        # For multiple references, compute similarity with each and take the maximum
        ref_similarities = []
        for reference in references:
            sim = compute_cosine_similarity(reference, hypothesis)
            ref_similarities.append(sim)
        
        # Take maximum similarity among all references
        max_similarity = max(ref_similarities) if ref_similarities else 0.0
        similarities.append(max_similarity)
    
    return similarities

def compute_enhanced_metrics(references, hypotheses, use_cosine_similarity=True):
    """Compute enhanced metrics including cosine similarity."""
    # Original metrics
    base_metrics = compute_metrics(references, hypotheses)
    
    # Add cosine similarity if requested
    if use_cosine_similarity:
        try:
            similarities = compute_cosine_similarities_batch(references, hypotheses)
            base_metrics['cosine_similarity'] = np.mean(similarities)
            base_metrics['cosine_similarity_std'] = np.std(similarities)
        except Exception as e:
            print(f"Warning: Could not compute cosine similarity: {e}")
            base_metrics['cosine_similarity'] = 0.0
            base_metrics['cosine_similarity_std'] = 0.0
    
    return base_metrics

def save_reference_indices(indices, save_path):
    with open(save_path, 'w') as f:
        json.dump(indices, f)

def load_reference_indices(save_path):
    if not os.path.exists(save_path):
        return None
    with open(save_path, 'r') as f:
        return json.load(f)

def select_reference_images(dataset, n=10, save_path='reference_indices.json', seed=42):
    random.seed(seed)
    indices = random.sample(range(len(dataset)), n)
    save_reference_indices(indices, save_path)
    return indices

def get_ground_truth_captions(dataset, idx):
    # For Flickr8k, there are multiple captions per image, grouped by image filename
    img_name = dataset.imgs[idx]
    # Find all captions for this image
    df = dataset.df
    caps = df[df['image'] == img_name]['caption'].tolist()
    return caps

def run_caption_comparison(models, dataset, indices, vocab, device, decoder_types):
    # models: dict {decoder_type: model}
    # indices: list of int (reference images)
    # decoder_types: list of str
    results = {dec: [] for dec in decoder_types}
    gts = []
    imgs = []
    for idx in indices:
        img, _ = dataset[idx]
        imgs.append(img)
        gt_caps = get_ground_truth_captions(dataset, idx)
        gts.append(gt_caps)
        for dec in decoder_types:
            model = models[dec]
            model.eval()
            with torch.no_grad():
                img_input = img.unsqueeze(0).to(device)
                features = model.encoder(img_input)
                if dec == 'attention':
                    # Optionally get attention weights (not implemented here)
                    caps = model.decoder.generate_caption(features, vocab=vocab)
                else:
                    caps = model.decoder.generate_caption(features, vocab=vocab)
            results[dec].append(' '.join(caps))
    return imgs, gts, results

def plot_caption_comparison(imgs, gts, results, decoder_types, output_dir=None):
    import os
    n = len(imgs)
    for i in range(n):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 1, 1)
        img = imgs[i]
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.axis('off')
        gt_text = '\n'.join(gts[i])
        title = f"GT:\n{gt_text}\n"
        for dec in decoder_types:
            title += f"{dec}: {results[dec][i]}\n"
        plt.title(title, fontsize=8)
        plt.tight_layout()
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"reference_{i+1}.png"))
            plt.close()
        plt.show()
        plt.close()


def evaluate_on_validation(model, val_loader, dataset, vocab, device, n_samples=None):
    references = []
    hypotheses = []
    for i, (imgs, caps) in enumerate(val_loader):
        imgs = imgs.to(device)
        for j in range(imgs.size(0)):
            img = imgs[j].unsqueeze(0)
            gt_caps = get_ground_truth_captions(dataset, i * val_loader.batch_size + j)
            references.append(gt_caps)
            with torch.no_grad():
                features = model.encoder(img)
                caps_pred = model.decoder.generate_caption(features, vocab=vocab)
                hypotheses.append(' '.join(caps_pred))
            if n_samples and len(hypotheses) >= n_samples:
                break
        if n_samples and len(hypotheses) >= n_samples:
            break
    return compute_metrics(references, hypotheses)
