import json
import numpy as np
import csv

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')
similarity_scores_dir = 'similarity_scores'

def load_text_embeddings(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Flatten all text embeddings into a dict: (image_name, idx) -> embedding
    text_embeddings = {}
    for image_name, v in data.items():
        for idx, emb in enumerate(v["text_embeddings"]):
            text_embeddings[(image_name, idx)] = np.array(emb)
    return text_embeddings

def load_image_embeddings(embeddings_path):
    """
    Load image embeddings from a specified path.
    This function assumes that the image embeddings are stored in a numpy array format.
    """
    with open(embeddings_path, 'r') as f:
        data = json.load(f)
    # Flatten all text embeddings into a dict: (image_name, idx) -> embedding
    image_embeddings = {}
    for image_name, v in data.items():
        image_embeddings[image_name] = np.array(v["image_embedding"])
    return image_embeddings

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def find_most_similar(image_name, image_embedding, text_embeddings):
    similarities = {}
    for (image_name, idx), text_emb in text_embeddings.items():
        sim = cosine_similarity(image_embedding, text_emb)
        similarities[(image_name, idx)] = sim
    most_similar = max(similarities, key=similarities.get)
    is_correct = (most_similar[0] == image_name)
    return most_similar, similarities, is_correct

def find_most_similar_text_for_image(all_text_keys, all_text_embs, image_embeddings):
    results = [] # shape: (N, D)
    for image_name, image_embedding in image_embeddings.items():
        print(f'Processing image: {image_name} for image-to-text similarity')
        img_emb = torch.tensor(image_embedding.astype(np.float32), device=device).unsqueeze(0)  # shape: (1, D)
        # Compute cosine similarity in parallel on GPU
        sim = torch.nn.functional.cosine_similarity(img_emb, all_text_embs)
        max_idx = torch.argmax(sim).item()
        most_similar = all_text_keys[max_idx]
        cosine_sim = sim[max_idx].item()
        is_correct = (most_similar[0] == image_name)
        results.append([image_name, most_similar[0], cosine_sim, is_correct])
        # Write results to CSV
    # CREATE A UUID FOR THE CSV FILE

    with open(similarity_scores_dir + '/' + 'img_to_text_similarity_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'most_similar_image_name', 'cosine_similarity', 'is_correct'])
        writer.writerows(results)

def find_most_similar_image_for_text(all_text_keys, all_text_embs, image_embeddings):
    # Get the most similar image for a text embedding
    text_to_image_results = []
    all_img_embs = []
    all_img_keys = []
    for img_name, img_emb in image_embeddings.items():
        all_img_embs.append(torch.tensor(img_emb.astype(np.float32), device=device))
        all_img_keys.append(img_name)
    all_img_embs = torch.stack(all_img_embs)

    for idx, (text_key, text_emb) in enumerate(zip(all_text_keys, all_text_embs)):
        print(f'Processing text: {text_key} for text-to-image similarity')
        text_emb = text_emb.unsqueeze(0)  # shape: (1, D)
        sim = torch.nn.functional.cosine_similarity(text_emb, all_img_embs)
        max_idx = torch.argmax(sim).item()
        most_similar_img = all_img_keys[max_idx]
        cosine_sim = sim[max_idx].item()
        is_correct = (text_key[0] == most_similar_img)
        text_to_image_results.append([text_key[0], text_key[1], most_similar_img, cosine_sim, is_correct])
    with open( similarity_scores_dir + '/' +'text_to_img_similarity_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text_image_name', 'text_image_idx', 'most_similar_image_name', 'cosine_similarity', 'is_correct'])
        writer.writerows(text_to_image_results)

def find_most_similar_text_for_text(all_text_keys, all_text_embs):
    # Get the most similar text for a text embedding (text-to-text)
    text_to_text_results = []
    for idx, (text_key, text_emb) in enumerate(zip(all_text_keys, all_text_embs)):
        print(f'Processing text: {text_key} for text-to-text similarity')
        text_emb = text_emb.unsqueeze(0)  # shape: (1, D)
        sim = torch.nn.functional.cosine_similarity(text_emb, all_text_embs)
        sim[idx] = -1.0  # Exclude self-similarity
        max_idx = torch.argmax(sim).item()
        most_similar_text = all_text_keys[max_idx]
        cosine_sim = sim[max_idx].item()
        is_image_correct = (text_key[0] == most_similar_text[0])
        is_text_correct = (text_key[0] == most_similar_text[0]) and (text_key[1] == most_similar_text[1])
        text_to_text_results.append([text_key[0], text_key[1], most_similar_text, cosine_sim, is_image_correct, is_text_correct])
    with open(similarity_scores_dir + '/' + 'text_to_text_similarity_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text_image', 'text_image_idx', 'most_similar_text_key', 'cosine_similarity', 'is_image_correct', 'is_text_correct'])
        writer.writerows(text_to_text_results)

def main():
    # Example usage
    text_embeddings = load_text_embeddings('embeddings.json')
    # Replace with actual image embedding loading
    image_embeddings = load_image_embeddings('embeddings.json')

    # Convert all text embeddings to a tensor on GPU
    all_text_embs = []
    all_text_keys = []
    for k, v in text_embeddings.items():
        all_text_embs.append(torch.tensor(v.astype(np.float32), device=device, dtype=torch.float32))
        all_text_keys.append(k)
    all_text_embs = torch.stack(all_text_embs)

    find_most_similar_text_for_image(all_text_keys, all_text_embs, image_embeddings)
    # Get the most similar image for a text embedding
    find_most_similar_image_for_text(all_text_keys, all_text_embs, image_embeddings)
    # Get the most similar text for a text embedding
    find_most_similar_text_for_text(all_text_keys, all_text_embs)


if __name__ == '__main__':
    main()