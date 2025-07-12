#!/usr/bin/env python
# coding: utf-8

"""
Dataset Download Script for Image Captioning Project

This script downloads and extracts the Flickr8k and Flickr30k datasets
from Google Drive links.
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm
import argparse
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

# Dataset configurations
DATASET_LINKS = {
    'flickr8k': {
        'url': 'https://drive.google.com/uc?export=download&id=1GBIRSf25OgXp1x3xs1g58M6TeGTEgRfl',
        'filename': 'flickr8k.zip',
        'size_mb': 1200,
        'description': 'Flickr8k dataset (~8,000 images, ~40,000 captions)'
    },
    'flickr30k': {
        'url': 'https://drive.google.com/uc?export=download&id=1uh5bZrfT4kbo3gJSWNjgmfc5ztVhJDmI',
        'filename': 'flickr30k.zip',
        'size_mb': 4500,
        'description': 'Flickr30k dataset (~31,000 images, ~155,000 captions)'
    }
}

def download_file(url, filename, expected_size_mb, file_id=None):
    """Download a file with progress bar. Uses gdown for Google Drive links if available."""
    print(f"Downloading {filename}...")
    print(f"Expected size: {expected_size_mb} MB")
    
    # Use gdown for Google Drive links if available
    if file_id and HAS_GDOWN:
        gdown.download(id=file_id, output=filename, quiet=False)
        return
    elif file_id and not HAS_GDOWN:
        print("gdown is not installed. Please install it with 'pip install gdown' for robust Google Drive downloads.")
        print("Falling back to requests, but download may fail for large files.")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        
        print(f"✓ Successfully downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {filename}: {str(e)}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file with progress indication."""
    print(f"Extracting {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files for progress indication
            total_files = len(zip_ref.namelist())
            print(f"Found {total_files} files to extract")
            
            zip_ref.extractall(extract_to)
        
        print(f"✓ Successfully extracted {zip_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error extracting {zip_path}: {str(e)}")
        return False

def verify_dataset_structure(dataset_type):
    """Verify that the dataset was extracted correctly."""
    print(f"Verifying {dataset_type} dataset structure...")
    
    if dataset_type == 'flickr8k':
        # Check for expected files
        expected_paths = [
            'resources/input/flickr8k/Images',
            'resources/input/flickr8k/captions.txt'
        ]
    elif dataset_type == 'flickr30k':
        expected_paths = [
            'resources/input/flickr30k/flickr30k_images/flickr30k_images',
            'resources/input/flickr30k/flickr30k_images/results.csv'
        ]
    else:
        print(f"✗ Unknown dataset type: {dataset_type}")
        return False
    
    all_exist = True
    for path in expected_paths:
        if os.path.exists(path):
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (missing)")
            all_exist = False
    
    if all_exist:
        print(f"✓ {dataset_type} dataset structure verified successfully!")
        return True
    else:
        print(f"✗ {dataset_type} dataset structure verification failed!")
        return False

def download_and_extract_dataset(dataset_type, force_download=False):
    """Download and extract a specific dataset."""
    if dataset_type not in DATASET_LINKS:
        print(f"Unknown dataset type: {dataset_type}")
        return False
    info = DATASET_LINKS[dataset_type]
    url = info['url']
    filename = os.path.join('resources', 'input', info['filename'])
    expected_size_mb = info['size_mb']
    description = info['description']
    file_id = None
    # Extract file id from url for gdown
    if 'id=' in url:
        file_id = url.split('id=')[-1]

    print("="*60)
    print(f"Processing {dataset_type.upper()} dataset")
    print(f"Description: {description}")
    print("="*60)

    # Download
    if not os.path.exists(filename) or force_download:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        download_file(url, filename, expected_size_mb, file_id=file_id)
    else:
        print(f"✓ {filename} already exists, skipping download.")

    # Extract
    print(f"Extracting {filename}...")
    try:
        extract_zip(filename, os.path.dirname(filename))
        print(f"✓ Successfully extracted {filename}")
    except Exception as e:
        print(f"✗ Error extracting {filename}: {e}")
        return False

    # Verify
        # Extract the file
        if not extract_zip(zip_path, "resources/input"):
            return False
    
    # Verify the structure
    if not verify_dataset_structure(dataset_type):
        return False
    
    return True

def main():
    """Main function to handle dataset downloads."""
    parser = argparse.ArgumentParser(description='Download and extract image captioning datasets')
    parser.add_argument('--dataset', choices=['flickr8k', 'flickr30k', 'both'], default='both',
                       help='Dataset to download (default: both)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download and re-extract even if files exist')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing datasets without downloading')
    
    args = parser.parse_args()
    
    print("Image Captioning Dataset Downloader")
    print("=" * 50)
    
    if args.verify_only:
        print("Verification mode - checking existing datasets...")
        datasets_to_check = ['flickr8k', 'flickr30k'] if args.dataset == 'both' else [args.dataset]
        
        for dataset_type in datasets_to_check:
            if verify_dataset_structure(dataset_type):
                print(f"✓ {dataset_type} dataset is ready to use!")
            else:
                print(f"✗ {dataset_type} dataset is not properly set up.")
        
        return
    
    # Determine which datasets to process
    if args.dataset == 'both':
        datasets_to_process = ['flickr8k', 'flickr30k']
    else:
        datasets_to_process = [args.dataset]
    
    success_count = 0
    total_count = len(datasets_to_process)
    
    for dataset_type in datasets_to_process:
        if download_and_extract_dataset(dataset_type, args.force):
            success_count += 1
        else:
            print(f"✗ Failed to process {dataset_type} dataset")
    
    print(f"\n{'='*60}")
    print(f"Download Summary: {success_count}/{total_count} datasets processed successfully")
    
    if success_count == total_count:
        print("✓ All datasets are ready to use!")
        print("\nYou can now run the image captioning training:")
        print("  python flickr_image_caption_with_pytorch_resnet_lstm.py --mode train --dataset flickr8k")
        print("  python flickr_image_caption_with_pytorch_resnet_lstm.py --mode train --dataset flickr30k")
    else:
        print("✗ Some datasets failed to download. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 