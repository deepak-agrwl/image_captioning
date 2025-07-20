#!/bin/bash

# Stop if any command fails
set -e

# Your file ID from Google Drive (make sure it's shareable)
FILE_ID="1uh5bZrfT4kbo3gJSWNjgmfc5ztVhJDmI"
ZIP_NAME="flickr30k.zip"

# Directory structure
ROOT_DIR="resources/input"
IMAGE_DIR="$ROOT_DIR/flickr30k_images"

echo "📥 Downloading Flickr30k dataset from Google Drive..."
mkdir -p "$ROOT_DIR"
gdown --id "$FILE_ID" -O "$ZIP_NAME"

echo "📦 Unzipping dataset..."
unzip -q "$ZIP_NAME" -d "$ROOT_DIR"

echo "📂 Organizing files..."
# Assuming the zip contains a folder like 'flickr30k_images' and 'results.csv' in root
# mv "$ROOT_DIR/flickr30k_images" "$IMAGE_DIR"
# mv "$ROOT_DIR/results.csv" "$ROOT_DIR/"

echo "🧹 Cleaning up..."
rm "$ZIP_NAME"

echo "✅ Setup complete. Dataset is ready at: $IMAGE_DIR"
