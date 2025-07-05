#!/usr/bin/env python
# coding: utf-8

"""
Test script to verify Flickr8k and Flickr30k dataset support
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flickr_image_caption_with_pytorch_resnet_lstm import create_data_loader, DATASET_CONFIGS

def test_dataset_loading():
    """Test loading both Flickr8k and Flickr30k datasets."""
    
    print("Testing dataset loading for both Flickr8k and Flickr30k...")
    print("=" * 60)
    
    for dataset_type in ['flickr8k', 'flickr30k']:
        print(f"\nTesting {dataset_type.upper()} dataset:")
        print("-" * 40)
        
        try:
            # Test data loader creation
            data_loader, dataset = create_data_loader(
                batch_size=2, 
                num_workers=0, 
                dataset_type=dataset_type
            )
            
            print(f"✓ Successfully created data loader for {dataset_type}")
            print(f"  - Dataset size: {len(dataset)}")
            print(f"  - Vocabulary size: {len(dataset.vocab)}")
            print(f"  - Number of batches: {len(data_loader)}")
            
            # Test loading a few samples
            print(f"  - Testing sample loading...")
            for i, (images, captions) in enumerate(data_loader):
                if i >= 2:  # Only test first 2 batches
                    break
                print(f"    Batch {i+1}: {images.shape}, {captions.shape}")
            
            print(f"✓ {dataset_type} dataset test completed successfully!")
            
        except Exception as e:
            print(f"✗ Error loading {dataset_type} dataset: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Dataset loading test completed!")

if __name__ == "__main__":
    test_dataset_loading() 