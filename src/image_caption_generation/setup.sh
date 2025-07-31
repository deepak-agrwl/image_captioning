#!/bin/bash
current_dir=$(basename "$PWD")
if [ "$current_dir" != "image_captioning" ]; then
  echo "Please execute this script from the 'image_captioning' directory, which is the root directory of this project."
  exit 1
fi

python -m venv venv
source venv/bin/activate
pip install -r src/image_caption_generation/requirements.txt

python src/image_caption_generation/download_datasets.py --dataset flickr30k

python -m spacy download en_core_web_sm
pip install --upgrade certifi

mkdir -p resources/models
if [ ! -f ./resources/models/resnet50-19c8e357.pth ]; then
  curl -k -o ./resources/models/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
fi