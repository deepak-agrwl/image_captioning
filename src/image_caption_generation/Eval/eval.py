#!/usr/bin/env python
# coding: utf-8

"""
Image Caption Generation Model Evaluation Script
===============================================

This script evaluates trained image captioning models using various metrics:
- BLEU-4 scores
- WER (Word Error Rate)
- ROUGE-L scores
- Cosine similarity between generated and reference captions

Usage:
    python eval.py --model_path path/to/model.pth --dataset_type flickr30k --eval_fraction 0.2
"""

import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules from the main project
from flickr_image_caption_with_pytorch_resnet_lstm import (
    Vocabulary, EncoderDecoder, CustomDataset, CapsCollate, DATASET_CONFIGS, load_trained_model
)
from experiment_utils import (
    compute_enhanced_metrics, initialize_embedding_model
)
import metrics_utils
from metrics_utils import compute_metrics

def get_ground_truth_captions_from_eval_dataset(eval_dataset, dataset_idx):
    """
    Get ground truth captions from evaluation dataset (handles both original dataset and Subset).
    
    Args:
        eval_dataset: Either CustomDataset or Subset object
        dataset_idx: Index within the evaluation dataset
    
    Returns:
        List of ground truth captions for the image
    """
    # Check if we're dealing with a Subset object
    if isinstance(eval_dataset, Subset):
        # Get the original dataset and the actual index
        original_dataset = eval_dataset.dataset
        actual_idx = eval_dataset.indices[dataset_idx]
        
        # Get image name from the original dataset
        img_name = original_dataset.imgs[actual_idx]
        
        # Find all captions for this image in the original dataset
        df = original_dataset.df
        caps = df[df['image'] == img_name]['caption'].tolist()
        
    else:
        # Working with original dataset directly
        img_name = eval_dataset.imgs[dataset_idx]
        df = eval_dataset.df
        caps = df[df['image'] == img_name]['caption'].tolist()
    
    return caps

def load_model_and_config(model_path, device):
    """Load the trained model and its configuration."""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model using the existing function
    model, vocab, checkpoint = load_trained_model(model_path, device)
    
    print(f"Model loaded successfully!")
    print(f"- Epoch: {checkpoint['epoch']}")
    print(f"- Training Loss: {checkpoint['training_loss']:.5f}")
    print(f"- Validation Loss: {checkpoint['validation_loss']:.5f}")
    print(f"- Vocabulary size: {len(vocab)}")
    
    # Extract hyperparameters
    hyperparams = checkpoint['hyperparameters']
    print(f"- Model configuration:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    return model, vocab, hyperparams, checkpoint

def load_evaluation_dataset(dataset_type, vocab, batch_size=32, num_workers=4, eval_fraction=1.0):
    """Load the evaluation dataset with optional fraction-based sampling."""
    print(f"Loading {dataset_type} dataset for evaluation...")
    
    # Define transforms
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    # Get dataset configuration
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_type]
    
    # Create full dataset
    full_dataset = CustomDataset(
        root_dir=config['image_dir'],
        captions_file=config['captions_file'],
        transform=transforms,
        dataset_type=dataset_type
    )
    
    # Use the loaded vocabulary instead of building a new one
    full_dataset.vocab = vocab
    
    # Apply fraction-based sampling if eval_fraction < 1.0
    if eval_fraction < 1.0:
        n_total = len(full_dataset)
        n_eval = int(eval_fraction * n_total)
        n_remaining = n_total - n_eval
        
        # Split dataset using random_split with fixed seed for reproducibility
        eval_dataset, _ = random_split(
            full_dataset, 
            [n_eval, n_remaining], 
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Full dataset: {n_total} samples")
        print(f"Evaluation subset: {n_eval} samples ({eval_fraction*100:.1f}%)")
    else:
        eval_dataset = full_dataset
        print(f"Using full dataset: {len(eval_dataset)} samples")

    # The vocab is already set on the original dataset
    # For Subset objects, we need to ensure the vocab is accessible
    if isinstance(eval_dataset, Subset):
        # Use the loaded vocabulary
        eval_dataset.vocab = vocab
    
    # Create data loader
    pad_idx = vocab.stoi["<PAD>"]
    data_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Don't shuffle for evaluation
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
    )
    
    return eval_dataset, data_loader

def generate_captions(model, eval_dataset, data_loader, device, max_samples=None):
    """Generate captions for all images in the evaluation dataset."""
    print("Generating captions...")
    
    model.eval()
    generated_captions = []
    reference_captions = []
    image_indices = []
    
    total_samples = 0
    max_samples = max_samples or len(eval_dataset)
    
    # Get vocabulary from the dataset
    if isinstance(eval_dataset, Subset):
        vocab = eval_dataset.dataset.vocab
    else:
        vocab = eval_dataset.vocab
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc="Generating captions")):
            images = images.to(device)
            batch_size = images.size(0)
            
            for i in range(batch_size):
                if total_samples >= max_samples:
                    break
                
                # Get single image
                img = images[i].unsqueeze(0)
                
                # Calculate dataset index within the evaluation dataset
                dataset_idx = batch_idx * data_loader.batch_size + i
                image_indices.append(dataset_idx)
                
                # Generate caption
                features = model.encoder(img)
                if hasattr(model.decoder, 'generate_caption'):
                    if hasattr(model.decoder, 'attention'):
                        # For attention decoder, check if it returns attention weights
                        try:
                            caps, _ = model.decoder.generate_caption(features, vocab=vocab, return_attention=True)
                        except:
                            caps = model.decoder.generate_caption(features, vocab=vocab)
                    else:
                        caps = model.decoder.generate_caption(features, vocab=vocab)
                else:
                    # Fallback for other decoder types
                    caps = model.decoder.generate_caption(features, vocab=vocab)
                
                generated_caption = ' '.join(caps)
                generated_captions.append(generated_caption)
                
                # Get ground truth captions using our new method
                try:
                    gt_caps = get_ground_truth_captions_from_eval_dataset(eval_dataset, dataset_idx)
                except Exception as e:
                    print(f"Warning: Could not get ground truth for sample {dataset_idx}: {e}")
                    # Fallback to a default caption
                    gt_caps = ["<no caption available>"]
                
                reference_captions.append(gt_caps)
                total_samples += 1
                
            if total_samples >= max_samples:
                break
    
    print(f"Generated {len(generated_captions)} captions")
    return generated_captions, reference_captions, image_indices

def evaluate_model(model_path, dataset_type, device='cpu', batch_size=32, num_workers=4, 
                  max_samples=None, use_cosine_similarity=True, output_dir=None, eval_fraction=1.0):
    """Main evaluation function."""
    print("=" * 60)
    print("IMAGE CAPTION GENERATION MODEL EVALUATION")
    print("=" * 60)
    
    # Load model and configuration
    model, vocab, hyperparams, checkpoint = load_model_and_config(model_path, device)
    
    # Initialize embedding model for cosine similarity
    if use_cosine_similarity:
        print("Initializing embedding model for cosine similarity...")
        try:
            # Try to use sentence-transformers model
            initialize_embedding_model(device=device)
        except Exception as e:
            print(f"Could not initialize transformer model: {e}")
            print("Will use fallback embedding method")
    
    # Load evaluation dataset with fraction-based sampling
    eval_dataset, data_loader = load_evaluation_dataset(dataset_type, vocab, batch_size, num_workers, eval_fraction)
    
    # Generate captions
    generated_captions, reference_captions, image_indices = generate_captions(
        model, eval_dataset, data_loader, device, max_samples
    )
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    if use_cosine_similarity:
        metrics = compute_enhanced_metrics(reference_captions, generated_captions, use_cosine_similarity=True)
    else:
        metrics = compute_metrics(reference_captions, generated_captions)
    
    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Dataset: {dataset_type}")
    print(f"Evaluation fraction: {eval_fraction*100:.1f}%")
    print(f"Samples evaluated: {len(generated_captions)}")
    print(f"Decoder type: {hyperparams.get('decoder_type', 'unknown')}")
    print(f"Encoder type: {hyperparams.get('encoder_type', 'resnet')}")
    print("-" * 40)
    print("METRICS:")
    print(f"  BLEU-4:        {metrics['bleu']:.4f}")
    print(f"  WER:           {metrics['wer']:.4f}")
    print(f"  ROUGE-L:       {metrics['rouge']:.4f}")
    if 'cosine_similarity' in metrics:
        print(f"  Cosine Sim:    {metrics['cosine_similarity']:.4f} ± {metrics['cosine_similarity_std']:.4f}")
    print("=" * 60)
    
    # Save detailed results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_detailed_results(
            model_path, dataset_type, hyperparams, metrics, 
            generated_captions, reference_captions, image_indices, output_dir, eval_fraction
        )
    
    return metrics, generated_captions, reference_captions

def save_detailed_results(model_path, dataset_type, hyperparams, metrics,
                         generated_captions, reference_captions, image_indices, output_dir, eval_fraction=1.0):
    """Save detailed results to files."""
    print(f"\nSaving detailed results to: {output_dir}")

    # Convert metrics and hyperparams to pandas Series for automatic type conversion
    metrics_series = pd.Series(metrics)
    hyperparams_series = pd.Series(hyperparams)

    # Save metrics summary
    summary = {
        'model_path': model_path,
        'dataset_type': dataset_type,
        'eval_fraction': eval_fraction,
        'hyperparameters': hyperparams_series.to_dict(),  # pandas handles numpy types
        'metrics': metrics_series.to_dict(),  # pandas handles numpy types
        'num_samples': len(generated_captions),
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Use pandas to_json() method which handles numpy types automatically
    summary_df = pd.Series(summary)
    summary_df.to_json(os.path.join(output_dir, 'evaluation_summary.json'), indent=2)

    # Save detailed caption comparisons
    results_data = []
    for i, (gen_cap, ref_caps, img_idx) in enumerate(zip(generated_captions, reference_captions, image_indices)):
        results_data.append({
            'sample_id': i,
            'image_index': img_idx,
            'generated_caption': gen_cap,
            'reference_captions': ref_caps,
            'num_references': len(ref_caps)
        })

    # Save as CSV for easy analysis
    df_results = pd.DataFrame([{
        'sample_id': item['sample_id'],
        'image_index': item['image_index'],
        'generated_caption': item['generated_caption'],
        'reference_caption_1': item['reference_captions'][0] if len(item['reference_captions']) > 0 else '',
        'reference_caption_2': item['reference_captions'][1] if len(item['reference_captions']) > 1 else '',
        'reference_caption_3': item['reference_captions'][2] if len(item['reference_captions']) > 2 else '',
        'reference_caption_4': item['reference_captions'][3] if len(item['reference_captions']) > 3 else '',
        'reference_caption_5': item['reference_captions'][4] if len(item['reference_captions']) > 4 else '',
        'num_references': item['num_references']
    } for item in results_data])

    df_results.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)

    # Use pandas DataFrame to_json() for the detailed results - handles numpy types automatically
    results_df = pd.DataFrame(results_data)
    results_df.to_json(os.path.join(output_dir, 'detailed_results.json'),
                       orient='records', indent=2)

    print(f"Results saved:")
    print(f"  - evaluation_summary.json")
    print(f"  - detailed_results.csv")
    print(f"  - detailed_results.json")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Evaluate Image Caption Generation Models")
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint (.pth file)')
    
    # Optional arguments
    parser.add_argument('--dataset_type', type=str, default='flickr30k',
                       choices=['flickr8k', 'flickr30k'],
                       help='Dataset type to evaluate on (default: flickr30k)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading (default: 4)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--eval_fraction', type=float, default=0.1,
                       help='Fraction of dataset to evaluate (0.0-1.0, default: 0.1)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for evaluation (default: auto)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save detailed results (default: none)')
    parser.add_argument('--no_cosine_similarity', action='store_true',
                       help='Skip cosine similarity computation')
    
    args = parser.parse_args()
    
    # Validate eval_fraction
    if not 0.0 < args.eval_fraction <= 1.0:
        print(f"Error: eval_fraction must be between 0.0 and 1.0, got {args.eval_fraction}")
        return 1
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Run evaluation
    try:
        metrics, generated_captions, reference_captions = evaluate_model(
            model_path=args.model_path,
            dataset_type=args.dataset_type,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            use_cosine_similarity=not args.no_cosine_similarity,
            output_dir=args.output_dir,
            eval_fraction=args.eval_fraction
        )
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
