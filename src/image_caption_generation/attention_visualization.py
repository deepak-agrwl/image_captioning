import matplotlib.pyplot as plt
import numpy as np
def visualize_attention(image, caption_tokens, attention_weights, vocab, figsize=(12, 8)):
    """
    Visualize attention weights over image for each generated token.
    image: torch.Tensor or numpy array (C x H x W or H x W x C)
    caption_tokens: list of tokens (words)
    attention_weights: list of attention maps (each is numpy array or torch tensor, shape: num_pixels)
    vocab: Vocabulary object (for <SOS>, <EOS> tokens)
    """
    if hasattr(image, 'cpu'):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    n_words = len(caption_tokens)
    n_cols = 5
    n_rows = int(np.ceil(n_words / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, word in enumerate(caption_tokens):
        row, col = divmod(i, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.imshow(image)
        attn = attention_weights[i]
        attn = attn.reshape(int(np.sqrt(attn.size)), -1) if attn.ndim == 1 else attn
        ax.imshow(attn, cmap='jet', alpha=0.5)
        ax.set_title(word)
        ax.axis('off')
    for j in range(i+1, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
