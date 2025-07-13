import os
import json
import random
import torch
import numpy as np
from metrics_utils import compute_metrics
from attention_visualization import visualize_attention
import matplotlib.pyplot as plt

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
