import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import csv
from tqdm import tqdm
import pandas as pd

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Configuration
image_embeddings_dir = Path("image_embeddings_pre")
caption_embeddings_dir = Path("caption_embeddings_pre")
similarity_scores_dir = Path("similarity_scores")
similarity_scores_dir.mkdir(exist_ok=True)

def load_image_embeddings():
    """Load all image embeddings from .pt files"""
    image_embeddings = {}
    print("Loading image embeddings...")
    
    for pt_file in tqdm(list(image_embeddings_dir.glob("*.pt")), desc="Loading image embeddings"):
        try:
            image_name = pt_file.stem  # Remove .pt extension
            embedding = torch.load(pt_file, map_location=device)
            # Convert to numpy array and flatten if needed
            if embedding.dim() > 1:
                embedding = embedding.squeeze()
            image_embeddings[image_name] = embedding.cpu().numpy()
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
    
    print(f"Loaded {len(image_embeddings)} image embeddings")
    return image_embeddings

def load_caption_embeddings():
    """Load all caption embeddings from .pt files"""
    caption_embeddings = {}
    print("Loading caption embeddings...")
    
    for pt_file in tqdm(list(caption_embeddings_dir.glob("*.pt")), desc="Loading caption embeddings"):
        try:
            # Parse filename: image_name_caption_index.pt
            filename = pt_file.stem
            parts = filename.split('_caption_')
            if len(parts) != 2:
                print(f"Skipping {filename}: unexpected format")
                continue
                
            image_name = parts[0]
            caption_index = parts[1]
            
            embedding = torch.load(pt_file, map_location=device)
            # Convert to numpy array and flatten if needed
            if embedding.dim() > 1:
                embedding = embedding.squeeze()
            
            caption_embeddings[(image_name, caption_index)] = embedding.cpu().numpy()
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
    
    print(f"Loaded {len(caption_embeddings)} caption embeddings")
    return caption_embeddings

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def calculate_image_to_caption_accuracy(image_embeddings, caption_embeddings, k_values=[1, 5, 10]):
    """Calculate Top-K accuracy for image-to-caption retrieval"""
    print("Calculating image-to-caption Top-K accuracy...")
    
    detailed_results = []
    top_k_results = {k: {'correct': 0, 'total': 0} for k in k_values}
    
    # Convert all caption embeddings to tensors for efficient computation
    all_caption_keys = list(caption_embeddings.keys())
    all_caption_embs = torch.stack([torch.tensor(caption_embeddings[key], dtype=torch.float32, device=device) 
                                   for key in all_caption_keys])
    
    for image_name, image_emb in tqdm(image_embeddings.items(), desc="Image-to-caption"):
        # Convert image embedding to tensor
        img_emb = torch.tensor(image_emb, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Calculate cosine similarity with all captions
        sim = F.cosine_similarity(img_emb, all_caption_embs)
        
        # Get top-k indices and values
        top_k_values, top_k_indices = torch.topk(sim, min(max(k_values), len(all_caption_keys)))
        
        # Store detailed results for all top-k matches
        for rank in range(len(top_k_indices)):
            caption_key = all_caption_keys[top_k_indices[rank]]
            cosine_sim = top_k_values[rank].item()
            is_correct = (caption_key[0] == image_name)
            rank_num = rank + 1
            
            detailed_results.append([
                image_name, 
                rank_num,
                caption_key[0], 
                caption_key[1], 
                cosine_sim, 
                is_correct
            ])
        
        # Check Top-K accuracy for each k value
        for k in k_values:
            if k <= len(top_k_indices):
                # Check if any of the top-k captions belong to the same image
                is_correct = any(all_caption_keys[idx][0] == image_name for idx in top_k_indices[:k])
                if is_correct:
                    top_k_results[k]['correct'] += 1
                top_k_results[k]['total'] += 1
    
    # Print Top-K results
    print("\nImage-to-Caption Top-K Accuracy Results:")
    for k in k_values:
        accuracy = top_k_results[k]['correct'] / top_k_results[k]['total'] if top_k_results[k]['total'] > 0 else 0
        print(f"Top-{k}: {accuracy:.4f} ({top_k_results[k]['correct']}/{top_k_results[k]['total']})")
    
    # Save detailed results to CSV (all top-k matches for every image)
    with open(similarity_scores_dir / 'image_to_caption_detailed_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'rank', 'caption_image_name', 'caption_index', 'cosine_similarity', 'is_correct'])
        writer.writerows(detailed_results)
    
    # Save Top-K summary results to CSV
    with open(similarity_scores_dir / 'image_to_caption_top_k_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['k', 'accuracy', 'correct', 'total'])
        for k in k_values:
            accuracy = top_k_results[k]['correct'] / top_k_results[k]['total'] if top_k_results[k]['total'] > 0 else 0
            writer.writerow([k, accuracy, top_k_results[k]['correct'], top_k_results[k]['total']])
    
    return top_k_results, detailed_results

def calculate_caption_to_image_accuracy(image_embeddings, caption_embeddings, k_values=[1, 5, 10]):
    """Calculate Top-K accuracy for caption-to-image retrieval"""
    print("Calculating caption-to-image Top-K accuracy...")
    
    detailed_results = []
    top_k_results = {k: {'correct': 0, 'total': 0} for k in k_values}
    
    # Convert all image embeddings to tensors for efficient computation
    all_image_keys = list(image_embeddings.keys())
    all_image_embs = torch.stack([torch.tensor(image_embeddings[key], dtype=torch.float32, device=device) 
                                 for key in all_image_keys])
    
    for caption_key, caption_emb in tqdm(caption_embeddings.items(), desc="Caption-to-image"):
        # Convert caption embedding to tensor
        cap_emb = torch.tensor(caption_emb, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Calculate cosine similarity with all images
        sim = F.cosine_similarity(cap_emb, all_image_embs)
        
        # Get top-k indices and values
        top_k_values, top_k_indices = torch.topk(sim, min(max(k_values), len(all_image_keys)))
        
        # Store detailed results for all top-k matches
        for rank in range(len(top_k_indices)):
            image_name = all_image_keys[top_k_indices[rank]]
            cosine_sim = top_k_values[rank].item()
            is_correct = (image_name == caption_key[0])
            rank_num = rank + 1
            
            detailed_results.append([
                caption_key[0], 
                caption_key[1],
                rank_num,
                image_name, 
                cosine_sim, 
                is_correct
            ])
        
        # Check Top-K accuracy for each k value
        for k in k_values:
            if k <= len(top_k_indices):
                # Check if any of the top-k images match the caption's image
                is_correct = any(all_image_keys[idx] == caption_key[0] for idx in top_k_indices[:k])
                if is_correct:
                    top_k_results[k]['correct'] += 1
                top_k_results[k]['total'] += 1
    
    # Print Top-K results
    print("\nCaption-to-Image Top-K Accuracy Results:")
    for k in k_values:
        accuracy = top_k_results[k]['correct'] / top_k_results[k]['total'] if top_k_results[k]['total'] > 0 else 0
        print(f"Top-{k}: {accuracy:.4f} ({top_k_results[k]['correct']}/{top_k_results[k]['total']})")
    
    # Save detailed results to CSV (all top-k matches for every caption)
    with open(similarity_scores_dir / 'caption_to_image_detailed_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['caption_image_name', 'caption_index', 'rank', 'image_name', 'cosine_similarity', 'is_correct'])
        writer.writerows(detailed_results)
    
    # Save Top-K summary results to CSV
    with open(similarity_scores_dir / 'caption_to_image_top_k_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['k', 'accuracy', 'correct', 'total'])
        for k in k_values:
            accuracy = top_k_results[k]['correct'] / top_k_results[k]['total'] if top_k_results[k]['total'] > 0 else 0
            writer.writerow([k, accuracy, top_k_results[k]['correct'], top_k_results[k]['total']])
    
    return top_k_results, detailed_results

def calculate_caption_to_caption_accuracy(caption_embeddings, k_values=[1, 5, 10]):
    """Calculate Top-K accuracy for caption-to-caption retrieval"""
    print("Calculating caption-to-caption Top-K accuracy...")
    
    detailed_results = []
    top_k_image_results = {k: {'correct': 0, 'total': 0} for k in k_values}
    top_k_exact_results = {k: {'correct': 0, 'total': 0} for k in k_values}
    
    # Convert all caption embeddings to tensors for efficient computation
    all_caption_keys = list(caption_embeddings.keys())
    all_caption_embs = torch.stack([torch.tensor(caption_embeddings[key], dtype=torch.float32, device=device) 
                                   for key in all_caption_keys])
    
    for idx, (caption_key, caption_emb) in enumerate(tqdm(caption_embeddings.items(), desc="Caption-to-caption")):
        # Convert caption embedding to tensor
        cap_emb = torch.tensor(caption_emb, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Calculate cosine similarity with all captions
        sim = F.cosine_similarity(cap_emb, all_caption_embs)
        sim[idx] = -1.0  # Exclude self-similarity
        
        # Get top-k indices and values
        top_k_values, top_k_indices = torch.topk(sim, min(max(k_values), len(all_caption_keys)))
        
        # Store detailed results for all top-k matches
        for rank in range(len(top_k_indices)):
            most_similar_caption = all_caption_keys[top_k_indices[rank]]
            cosine_sim = top_k_values[rank].item()
            is_image_correct = (most_similar_caption[0] == caption_key[0])
            is_caption_correct = (most_similar_caption[0] == caption_key[0]) and (most_similar_caption[1] == caption_key[1])
            rank_num = rank + 1
            
            detailed_results.append([
                caption_key[0], 
                caption_key[1],
                rank_num,
                most_similar_caption[0], 
                most_similar_caption[1], 
                cosine_sim, 
                is_image_correct, 
                is_caption_correct
            ])
        
        # Check Top-K accuracy for each k value
        for k in k_values:
            if k <= len(top_k_indices):
                # Check if any of the top-k captions belong to the same image
                is_image_correct = any(all_caption_keys[idx][0] == caption_key[0] for idx in top_k_indices[:k])
                # Check if any of the top-k captions are the exact same caption
                is_exact_correct = any((all_caption_keys[idx][0] == caption_key[0] and 
                                       all_caption_keys[idx][1] == caption_key[1]) for idx in top_k_indices[:k])
                
                if is_image_correct:
                    top_k_image_results[k]['correct'] += 1
                if is_exact_correct:
                    top_k_exact_results[k]['correct'] += 1
                
                top_k_image_results[k]['total'] += 1
                top_k_exact_results[k]['total'] += 1
    
    # Print Top-K results
    print("\nCaption-to-Caption Top-K Accuracy Results:")
    print("Image-level accuracy (same image):")
    for k in k_values:
        accuracy = top_k_image_results[k]['correct'] / top_k_image_results[k]['total'] if top_k_image_results[k]['total'] > 0 else 0
        print(f"Top-{k}: {accuracy:.4f} ({top_k_image_results[k]['correct']}/{top_k_image_results[k]['total']})")
    
    print("\nExact match accuracy (same image and caption index):")
    for k in k_values:
        accuracy = top_k_exact_results[k]['correct'] / top_k_exact_results[k]['total'] if top_k_exact_results[k]['total'] > 0 else 0
        print(f"Top-{k}: {accuracy:.4f} ({top_k_exact_results[k]['correct']}/{top_k_exact_results[k]['total']})")
    
    # Save detailed results to CSV (all top-k matches for every caption)
    with open(similarity_scores_dir / 'caption_to_caption_detailed_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['caption_image_name', 'caption_index', 'rank', 'most_similar_image_name', 'most_similar_caption_index', 
                        'cosine_similarity', 'is_image_correct', 'is_caption_correct'])
        writer.writerows(detailed_results)
    
    # Save Top-K summary results to CSV
    with open(similarity_scores_dir / 'caption_to_caption_top_k_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['k', 'image_accuracy', 'image_correct', 'image_total', 'exact_accuracy', 'exact_correct', 'exact_total'])
        for k in k_values:
            image_accuracy = top_k_image_results[k]['correct'] / top_k_image_results[k]['total'] if top_k_image_results[k]['total'] > 0 else 0
            exact_accuracy = top_k_exact_results[k]['correct'] / top_k_exact_results[k]['total'] if top_k_exact_results[k]['total'] > 0 else 0
            writer.writerow([k, image_accuracy, top_k_image_results[k]['correct'], top_k_image_results[k]['total'],
                           exact_accuracy, top_k_exact_results[k]['correct'], top_k_exact_results[k]['total']])
    
    return top_k_image_results, top_k_exact_results, detailed_results

def main():
    print("Loading embeddings...")
    
    # Load embeddings
    image_embeddings = load_image_embeddings()
    caption_embeddings = load_caption_embeddings()
    
    if not image_embeddings or not caption_embeddings:
        print("No embeddings found. Please run generate_embeddings.py first.")
        return
    
    print(f"\nLoaded {len(image_embeddings)} image embeddings and {len(caption_embeddings)} caption embeddings")
    
    # Calculate accuracies
    print("\n" + "="*60)
    print("CALCULATING TOP-K ACCURACY METRICS")
    print("="*60)
    
    # Image-to-caption accuracy
    img_to_cap_top_k_results, img_to_cap_detailed_results = calculate_image_to_caption_accuracy(image_embeddings, caption_embeddings)
    
    # Caption-to-image accuracy
    cap_to_img_top_k_results, cap_to_img_detailed_results = calculate_caption_to_image_accuracy(image_embeddings, caption_embeddings)
    
    # Caption-to-caption accuracy
    cap_to_cap_img_top_k_results, cap_to_cap_exact_top_k_results, cap_to_cap_detailed_results = calculate_caption_to_caption_accuracy(caption_embeddings)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("Image-to-Caption Top-K Accuracy:")
    for k in [1, 5, 10]:
        accuracy = img_to_cap_top_k_results[k]['correct'] / img_to_cap_top_k_results[k]['total'] if img_to_cap_top_k_results[k]['total'] > 0 else 0
        print(f"  Top-{k}: {accuracy:.4f} ({img_to_cap_top_k_results[k]['correct']}/{img_to_cap_top_k_results[k]['total']})")
    
    print("\nCaption-to-Image Top-K Accuracy:")
    for k in [1, 5, 10]:
        accuracy = cap_to_img_top_k_results[k]['correct'] / cap_to_img_top_k_results[k]['total'] if cap_to_img_top_k_results[k]['total'] > 0 else 0
        print(f"  Top-{k}: {accuracy:.4f} ({cap_to_img_top_k_results[k]['correct']}/{cap_to_img_top_k_results[k]['total']})")
    
    print("\nCaption-to-Caption Top-K Accuracy (Image-level):")
    for k in [1, 5, 10]:
        accuracy = cap_to_cap_img_top_k_results[k]['correct'] / cap_to_cap_img_top_k_results[k]['total'] if cap_to_cap_img_top_k_results[k]['total'] > 0 else 0
        print(f"  Top-{k}: {accuracy:.4f} ({cap_to_cap_img_top_k_results[k]['correct']}/{cap_to_cap_img_top_k_results[k]['total']})")
    
    print("\nCaption-to-Caption Top-K Accuracy (Exact match):")
    for k in [1, 5, 10]:
        accuracy = cap_to_cap_exact_top_k_results[k]['correct'] / cap_to_cap_exact_top_k_results[k]['total'] if cap_to_cap_exact_top_k_results[k]['total'] > 0 else 0
        print(f"  Top-{k}: {accuracy:.4f} ({cap_to_cap_exact_top_k_results[k]['correct']}/{cap_to_cap_exact_top_k_results[k]['total']})")
    
    # Save summary to file
    with open(similarity_scores_dir / 'accuracy_summary.txt', 'w') as f:
        f.write("TOP-K ACCURACY SUMMARY\n")
        f.write("="*40 + "\n\n")
        
        f.write("Image-to-Caption Top-K Accuracy:\n")
        for k in [1, 5, 10]:
            accuracy = img_to_cap_top_k_results[k]['correct'] / img_to_cap_top_k_results[k]['total'] if img_to_cap_top_k_results[k]['total'] > 0 else 0
            f.write(f"  Top-{k}: {accuracy:.4f} ({img_to_cap_top_k_results[k]['correct']}/{img_to_cap_top_k_results[k]['total']})\n")
        
        f.write("\nCaption-to-Image Top-K Accuracy:\n")
        for k in [1, 5, 10]:
            accuracy = cap_to_img_top_k_results[k]['correct'] / cap_to_img_top_k_results[k]['total'] if cap_to_img_top_k_results[k]['total'] > 0 else 0
            f.write(f"  Top-{k}: {accuracy:.4f} ({cap_to_img_top_k_results[k]['correct']}/{cap_to_img_top_k_results[k]['total']})\n")
        
        f.write("\nCaption-to-Caption Top-K Accuracy (Image-level):\n")
        for k in [1, 5, 10]:
            accuracy = cap_to_cap_img_top_k_results[k]['correct'] / cap_to_cap_img_top_k_results[k]['total'] if cap_to_cap_img_top_k_results[k]['total'] > 0 else 0
            f.write(f"  Top-{k}: {accuracy:.4f} ({cap_to_cap_img_top_k_results[k]['correct']}/{cap_to_cap_img_top_k_results[k]['total']})\n")
        
        f.write("\nCaption-to-Caption Top-K Accuracy (Exact match):\n")
        for k in [1, 5, 10]:
            accuracy = cap_to_cap_exact_top_k_results[k]['correct'] / cap_to_cap_exact_top_k_results[k]['total'] if cap_to_cap_exact_top_k_results[k]['total'] > 0 else 0
            f.write(f"  Top-{k}: {accuracy:.4f} ({cap_to_cap_exact_top_k_results[k]['correct']}/{cap_to_cap_exact_top_k_results[k]['total']})\n")
    
    print(f"\nResults saved to {similarity_scores_dir}/")

if __name__ == "__main__":
    main() 