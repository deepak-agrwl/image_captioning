#!/usr/bin/env python
# coding: utf-8

"""
Demonstration script for model evaluation
=========================================

This script demonstrates how to use the evaluation functionality 
with the example model provided.

Usage:
    python demo_eval.py
"""

import os
import sys

def main():
    """Demonstrate the evaluation functionality."""
    print("Image Caption Generation Model Evaluation Demo")
    print("=" * 50)
    
    # Example model path (the one provided in the request)
    model_path = "/Users/deagrawa/github/image_captioning/resources/output/flickr30k_attention_ep25_bs128_lr1e4_cosine_att_20250712_164848/flickr30k/resnet_attention/best_model.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Example model not found at: {model_path}")
        print("\nTo use this demo, either:")
        print("1. Ensure the example model exists at the above path, or")
        print("2. Modify the model_path variable to point to your trained model")
        return 1
    
    print(f"Using model: {model_path}")
    print(f"Dataset: flickr30k")
    print("\nRunning evaluation...")
    
    # Command to run the evaluation
    cmd = f"""python eval.py \\
    --model_path "{model_path}" \\
    --dataset_type flickr30k \\
    --batch_size 16 \\
    --num_workers 4 \\
    --max_samples 100 \\
    --output_dir ./evaluation_results \\
    --device auto"""
    
    print(f"\nCommand being executed:")
    print(cmd)
    print("\n" + "=" * 50)
    
    # Execute the command
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("Check the './evaluation_results' directory for detailed results.")
    else:
        print(f"\nDemo failed with exit code: {exit_code}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 