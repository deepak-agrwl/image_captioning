# Multi-Dataset Image Captioning with PyTorch ResNet-LSTM

This project implements an image captioning model using ResNet50 as the image encoder and LSTM as the text decoder, supporting both Flickr8k and Flickr30k datasets.

## Overview

The model architecture consists of:
- **Encoder**: ResNet50 (pretrained) for image feature extraction
- **Decoder**: LSTM network for generating captions
- **Supported Datasets**: Flickr8k (8,000 images) and Flickr30k (31,000 images)

## Dataset Downloads

The datasets are not included in the git repository due to their large size. You need to download them separately from Google Drive:

### Flickr8k Dataset
- **Download Link**: [flickr8k.zip](https://drive.google.com/file/d/1GBIRSf25OgXp1x3xs1g58M6TeGTEgRfl/view?usp=drive_link)
- **File Size**: ~1.2 GB
- **Contents**: Images and captions for ~8,000 images

### Flickr30k Dataset
- **Download Link**: [flickr30k.zip](https://drive.google.com/file/d/1uh5bZrfT4kbo3gJSWNjgmfc5ztVhJDmI/view?usp=drive_link)
- **File Size**: ~4.5 GB
- **Contents**: Images and captions for ~31,000 images

## Prerequisites

### 1. Python Environment
- Python 3.9 or higher
- pip package manager

### 2. Required Python Packages
Install the required packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

### 3. spaCy English Model
Download the spaCy English language model:
```bash
python -m spacy download en_core_web_sm
```

### 4. Pretrained ResNet50 Model
Download the pretrained ResNet50 weights:
```bash
# Create the models directory
mkdir -p ../../resources/models

# Download the pretrained model
curl -k -o ../../resources/models/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## Setup Instructions

### Option 1: Automated Download (Recommended)
Use the provided download script for easy setup:

```bash
# Install all dependencies (including download script dependencies)
pip install -r requirements.txt

# Download and extract both datasets
python download_datasets.py

# Or download specific datasets
python download_datasets.py --dataset flickr8k
python download_datasets.py --dataset flickr30k

# Verify existing datasets without downloading
python download_datasets.py --verify-only

# Force re-download (if files are corrupted)
python download_datasets.py --force
```

### Option 2: Manual Download
If you prefer to download manually:

```bash
# Create the resources directory structure
mkdir -p resources/input

# Download datasets (you can also download manually from the links above)
# For Flickr8k:
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1GBIRSf25OgXp1x3xs1g58M6TeGTEgRfl' -O resources/input/flickr8k.zip

# For Flickr30k:
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1uh5bZrfT4kbo3gJSWNjgmfc5ztVhJDmI' -O resources/input/flickr30k.zip

# Extract datasets
cd resources/input
unzip flickr8k.zip
unzip flickr30k.zip

# Verify the directory structure
ls -la flickr8k/
ls -la flickr30k/
```

### 3. Verify Dataset Structure
After extraction, you should have the following structure:

```
resources/input/
├── flickr8k/
│   ├── Images/
│   │   ├── 1000268201_693b08cb0e.jpg
│   │   ├── 1001773457_577c3a7d70.jpg
│   │   └── ... (8,000+ images)
│   └── captions.txt
└── flickr30k/
    └── flickr30k_images/
        ├── flickr30k_images/
        │   ├── 1000092795.jpg
        │   ├── 10002456.jpg
        │   └── ... (31,000+ images)
        └── captions.csv
```

## Quick Setup

Run the setup script to install dependencies and download required files:
```bash
chmod +x setup.sh
./setup.sh
```

## Supported Datasets

### Flickr8k Dataset
- **Images**: `resources/input/flickr8k/Images/`
- **Captions**: `resources/input/flickr8k/captions.txt`
- **Format**: CSV with columns `image,caption`
- **Size**: ~8,000 images, ~40,000 captions

### Flickr30k Dataset
- **Images**: `resources/input/flickr30k/flickr30k_images/flickr30k_images/`
- **Captions**: `resources/input/flickr30k/flickr30k_images/captions.csv`
- **Format**: CSV with columns `image,caption_number,caption,id`
- **Size**: ~31,000 images, ~155,000 captions

## Usage

### Training & Evaluation

Train a model using the provided script:

```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py --mode train --dataset flickr8k --decoder attention --epochs 10
```

- Model checkpoints are saved after each epoch in `saved_models/<dataset>/`.
- The best model (lowest validation loss so far) is always tracked and saved as `best_model.pth`.
- Loss curves (as PNG) and metrics history (as JSON) are saved after each epoch for easy experiment tracking.
- All training, validation, and metric history is tracked and saved automatically.

### Resume Training from Checkpoint

To resume training for N more epochs from a checkpoint:

```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py --mode train --model_path saved_models/flickr8k/model_epoch5.pth --epochs 5 --dataset flickr8k --decoder attention
```

- This will load the checkpoint and continue training for 5 more epochs (not up to a total).
- All optimizer/scheduler state is restored for seamless continuation.

## Code Structure & Modularity

- `flickr_image_caption_with_pytorch_resnet_lstm.py`: Main training/testing script (now modular, delegates all saving/checkpoint/metrics to utility modules)
- `training_utils.py`: Handles saving model checkpoints, best model, metrics, and loss curves
- `metrics_utils.py`, `experiment_utils.py`, `attention_visualization.py`: Metrics, experiment management, and visualization utilities

## Automated Experiment Tracking

- Model checkpoints, best model, loss curves, and metrics are all saved after each epoch
- No manual checkpoint code is needed in the main script—everything is modular and handled by utility modules
- All progress is tracked and can be resumed or analyzed at any time

## Example Resume Training

To continue training from a previous run:

```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py --mode train --model_path saved_models/flickr8k/model_epoch10.pth --epochs 5 --dataset flickr8k --decoder attention
```

This will load the model, optimizer, and scheduler state from the checkpoint and continue for 5 more epochs, saving all progress as before.

## Command Line Arguments

- `--mode`: Choose between 'train' or 'test' (default: 'train')
- `--dataset`: Choose between 'flickr8k' or 'flickr30k' (default: 'flickr8k')
- `--decoder`: Choose between 'lstm' (default) or 'attention' (LSTM+Attention)
- `--model_path`: Path to saved model for testing
- `--epochs`: Number of training epochs (default: 10)
- `--model_path`: Path to saved model for testing
- `--epochs`: Number of training epochs (default: 10)

## Configuration

You can modify the following parameters in the script:

```python
# Model hyperparameters
embed_size = 400          # Embedding dimension
hidden_size = 512         # LSTM hidden size
num_layers = 2            # Number of LSTM layers
batch_size = 32           # Training batch size
num_epochs = 2            # Number of training epochs
learning_rate = 0.0002    # Learning rate (scaled for batch size)
```

## Model Architecture

### Encoder (ResNet50)
- Extracts 2048-dimensional features from images
- Projects to 400-dimensional embedding space
- Parameters are frozen during training

### Decoder (LSTM & LSTM+Attention)
- **LSTM**: Takes image features and generates captions word by word (default)
- **LSTM+Attention**: Optionally, use an attention mechanism on top of LSTM for potentially improved captioning performance. Select with `--decoder attention`.
- Outputs probability distribution over vocabulary

### Vocabulary
- Built from training captions with frequency threshold
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Typical size: ~3000-4000 words for Flickr8k, ~8000-10000 for Flickr30k

## Training Process

The script will:
1. Load and preprocess the selected dataset (Flickr8k or Flickr30k)
2. Build vocabulary from captions
3. Initialize the ResNet50-LSTM model
4. Train the model with progress bars
5. Generate sample captions during training
6. Save models in dataset-specific directories
7. Test the model on random images

## Output

### Training Progress
- Real-time progress bars showing epoch and batch progress
- Current loss and average loss per epoch
- Sample caption generation every epoch
- Loss plots saved with dataset information

### Sample Output
```
Flickr Image Captioning with PyTorch ResNet-LSTM
==================================================
Using device: mps
Using dataset: flickr8k
Creating data loader...
Dataset size: 40455
Vocabulary size: 2994
Initializing model...
Model parameters: 23,512,000

Starting training for 10 epochs on flickr8k dataset...
Epoch 1/10: 100%|██████████| 1264/1264 [15:30<00:00, 1.36batch/s, Loss=2.12345, Avg Loss=2.45678]

Epoch 1 - Evaluating intermediate model...
Sample caption: a man is playing with a dog in the park
Epoch 1 completed:
  Training Loss: 2.45678
  Validation Loss: 2.34567
  Model saved: model_epoch_01_loss_2.4568_20241201_143022.pt
```

## Testing

### Test Dataset Loading
Run the test script to verify dataset loading:
```bash
python test_dataset_support.py
```

This will test loading both datasets and report any issues.

### Quick Dataset Verification
```bash
# Check if datasets are properly loaded
python -c "
from flickr_image_caption_with_pytorch_resnet_lstm import create_data_loader
data_loader, dataset = create_data_loader(batch_size=2, dataset_type='flickr8k')
print(f'Flickr8k: {len(dataset)} samples, {len(dataset.vocab)} vocabulary')
data_loader, dataset = create_data_loader(batch_size=2, dataset_type='flickr30k')
print(f'Flickr30k: {len(dataset)} samples, {len(dataset.vocab)} vocabulary')
"
```

## Download Script Features

The `download_datasets.py` script provides several useful features:

### Basic Usage
```bash
# Download both datasets
python download_datasets.py

# Download specific dataset
python download_datasets.py --dataset flickr8k
```

### Advanced Options
```bash
# Verify existing datasets without downloading
python download_datasets.py --verify-only

# Force re-download (useful if files are corrupted)
python download_datasets.py --force

# Download specific dataset with force option
python download_datasets.py --dataset flickr30k --force
```

### Features
- **Progress bars**: Shows download and extraction progress
- **Resume capability**: Skips download if file already exists
- **Verification**: Checks dataset structure after extraction
- **Error handling**: Provides clear error messages
- **Flexible options**: Download one or both datasets

## Dataset Statistics

### Flickr8k
- ~8,000 images
- ~40,000 captions (5 per image)
- Smaller vocabulary size
- Faster training time

### Flickr30k
- ~31,000 images
- ~155,000 captions (5 per image)
- Larger vocabulary size
- Longer training time

## Performance Considerations

### Memory Requirements
- **Flickr8k**: ~4GB RAM recommended
- **Flickr30k**: ~8GB RAM recommended

### Training Time Estimates
- **Flickr8k**: ~30 mins for 10 epochs (MacOS M4 Max GPU (40 Cores))
- **Flickr30k**: ~2 hours for 10 epochs (MacOS M4 Max GPU (40 Cores))

### Recommended Settings
```bash
# For Flickr8k (faster training)
python flickr_image_caption_with_pytorch_resnet_lstm.py --mode train --dataset flickr8k --epochs 15

# For Flickr30k (longer training, better results)
python flickr_image_caption_with_pytorch_resnet_lstm.py --mode train --dataset flickr30k --epochs 10
```

## Model Performance

Typical training metrics:
- **Loss**: Starts ~4.0, converges to ~2.0-2.5
- **Training Time**: 
  - Flickr8k: ~15-30 minutes per epoch (depending on hardware)
  - Flickr30k: ~2-4 hours per epoch (depending on hardware)
- **Memory Usage**: ~4-8 GB GPU memory with batch_size=32

## Key Changes Made

### 1. Dataset Configuration
- Added `DATASET_CONFIGS` dictionary to store dataset-specific paths and settings
- Each dataset has its own image directory, captions file, and column mappings

### 2. CustomDataset Class Updates
- Added `dataset_type` parameter to handle different file formats
- Supports both Flickr8k (.txt) and Flickr30k (.csv) caption files
- Automatically handles different column structures

### 3. Model Organization
- Models are now saved in dataset-specific subdirectories:
  - `saved_models/flickr8k/` for Flickr8k models
  - `saved_models/flickr30k/` for Flickr30k models
- Training info and loss plots are saved with dataset information

### 4. Function Updates
- `create_data_loader()`: Now accepts `dataset_type` parameter
- `train_model()`: Saves models in dataset-specific directories
- `test_model()`: Uses correct dataset for testing
- `create_loss_plots()`: Includes dataset information in plot titles

## Troubleshooting

### Common Issues

1. **Dataset not found error**:
   - Ensure you've downloaded and extracted the datasets correctly
   - Check the directory structure matches the expected paths

2. **Memory errors during training**:
   - Reduce batch size: modify the `batch_size` parameter in the code
   - Use gradient accumulation for larger effective batch sizes

3. **Slow training**:
   - Use GPU if available (CUDA or MPS)
   - Reduce number of workers if using CPU

4. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **CUDA out of memory**
   - Reduce batch size: `batch_size = 16`
   - Reduce model size: `embed_size = 256`

6. **Pretrained model not found**
   ```bash
   curl -k -o ../../resources/models/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
   ```

### File Structure Verification
```bash
# Verify Flickr8k structure
ls resources/input/flickr8k/Images/ | wc -l  # Should be ~8,000
head -5 resources/input/flickr8k/captions.txt

# Verify Flickr30k structure  
ls resources/input/flickr30k/flickr30k_images/flickr30k_images/ | wc -l  # Should be ~31,000
head -5 resources/input/flickr30k/flickr30k_images/captions.csv
```

### Performance Tips

- **GPU Training**: The script automatically detects and uses CUDA/MPS if available
- **Memory Optimization**: Adjust batch size based on your GPU memory
- **Data Loading**: Increase `num_workers` for faster data loading (if memory allows)

## File Structure

```
src/image_caption_generation/
├── README_final.md                                    # This comprehensive guide
├── flickr_image_caption_with_pytorch_resnet_lstm.py   # Main training script
├── download_datasets.py                               # Dataset download script
├── test_dataset_support.py                            # Dataset testing script
├── requirements.txt                                   # Python dependencies (includes download script deps)
├── setup.sh                                           # Setup script
├── test_request_ssl.py                                # SSL test utility
└── flickr_image_caption_with_pytorch_resnet_lstm.ipynb # Jupyter notebook version - Update on the initial one to make it working
```

## Next Steps

- [Done] Add code to save the loss and accuracy metrics to a file + Add code to generate loss graph
- [Done] Enhance it to work with 30K Image Captions set
- Add code to generate accuracy of the model
- [In Progress] Train the model for 30K image set on GPU machine in runpod.io and retrieve the trained model, loss graph
- Enhance Decoder with Attention Mechanism
- Enhance model by replacing LSTM with ViT model.
- Enhance Decoder model by replacing it with Transformer Based Model

## Customization

### Adding New Datasets
1. Add dataset configuration to `DATASET_CONFIGS` dictionary
2. Ensure captions are in a supported format
3. Adjust vocabulary frequency threshold if needed

### Model Modifications
- Change encoder: Modify `EncoderCNN` class
- Change decoder: Modify `DecoderRNN` class
- Add attention: Implement attention mechanism in decoder

## Notes

- The vocabulary is built separately for each dataset
- Models trained on one dataset cannot be directly used on the other due to different vocabularies
- Training on Flickr30k will take longer due to the larger dataset size
- Consider adjusting batch size and learning rate for different dataset sizes
- The datasets are not included in git due to size constraints - download them separately

## License

This implementation is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests! 