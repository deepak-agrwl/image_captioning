# Flickr8k Image Captioning with PyTorch ResNet-LSTM

This folder contains the implementation of an image captioning model using ResNet50 as the image encoder and LSTM as the text decoder, trained on the Flickr8k dataset.

## Overview

The model architecture consists of:
- **Encoder**: ResNet50 (pretrained) for image feature extraction
- **Decoder**: LSTM network for generating captions
- **Dataset**: Flickr8k (8,000 images with 5 captions each)

## Prerequisites

### 1. Python Environment
- Python 3.7 or higher
- pip package manager

### 2. Required Python Packages
Install the required packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision
pip install spacy
pip install pandas numpy matplotlib
pip install tqdm pillow
```

### 3. spaCy English Model
Download the spaCy English language model:
```bash
python -m spacy download en_core_web_sm
```

### 4. Dataset Setup
Ensure you have the Flickr8k dataset in the correct location:
```
resources/input/flickr8k/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   ├── 1001773457_577c3a7d70.jpg
│   └── ... (8,000 images)
└── captions.txt
```

### 5. Pretrained ResNet50 Model
Download the pretrained ResNet50 weights:
```bash
# Create the models directory
mkdir -p ../../resources/models

# Download the pretrained model
curl -k -o ../../resources/models/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## Quick Setup

Run the setup script to install dependencies and download required files:
```bash
chmod +x setup.sh
./setup.sh
```

## Usage

### Training the Model

Run the main training script:
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py
```

### Configuration

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

### Training Process

The script will:
1. Load and preprocess the Flickr8k dataset
2. Build vocabulary from captions
3. Initialize the ResNet50-LSTM model
4. Train the model with progress bars
5. Generate sample captions during training
6. Test the model on random images

## Model Architecture

### Encoder (ResNet50)
- Extracts 2048-dimensional features from images
- Projects to 400-dimensional embedding space
- Parameters are frozen during training

### Decoder (LSTM)
- Takes image features and generates captions word by word
- Outputs probability distribution over vocabulary

### Vocabulary
- Built from training captions with frequency threshold
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Typical size: ~3000-4000 words

## Output

### Training Progress
- Real-time progress bars showing epoch and batch progress
- Current loss and average loss per epoch
- Sample caption generation every 1000 batches

### Sample Output
```
Flickr Image Captioning with PyTorch ResNet-LSTM
==================================================
Using device: cuda
Creating data loader...
Dataset size: 40455
Vocabulary size: 2994
Initializing model...
Model parameters: 23,512,000

Starting training...
Epoch 1/2: 100%|██████████| 1264/1264 [15:30<00:00, 1.36batch/s, Loss=2.12345, Avg Loss=2.45678]

Epoch 1 - Evaluating intermediate model...
Sample caption: a man is playing with a dog in the park
Epoch 1 completed - Average Loss: 2.45678
```

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **CUDA out of memory**
   - Reduce batch size: `batch_size = 16`
   - Reduce model size: `embed_size = 256`

3. **Pretrained model not found**
   ```bash
   curl -k -o ../../resources/models/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
   ```

4. **Dataset not found**
   - Ensure Flickr8k dataset is in `resources/input/flickr8k/`
   - Check file paths in the script

### Performance Tips

- **GPU Training**: The script automatically detects and uses CUDA/MPS if available
- **Memory Optimization**: Adjust batch size based on your GPU memory
- **Data Loading**: Increase `num_workers` for faster data loading (if memory allows)

## File Structure

```
src/image_caption_generation/
├── README.md                                    # This file
├── flickr_image_caption_with_pytorch_resnet_lstm.py  # Main training script
├── requirements.txt                             # Python dependencies
├── setup.sh                                     # Setup script
├── test_request_ssl.py                          # SSL test utility
└── flickr_image_caption_with_pytorch_resnet_lstm.ipynb  # Jupyter notebook version
```

## Model Performance

Typical training metrics:
- **Loss**: Starts ~4.0, converges to ~2.0-2.5
- **Training Time**: ~15-30 minutes per epoch (depending on hardware)
- **Memory Usage**: ~4-8 GB GPU memory with batch_size=32

## Next Steps

- [Done] Add code to save the loss and accuracy metrics to a file + Add code to generate loss graph
- Add code to generate accuracy of the model. 
- Enhance it to work with 30K Image Captions set. Try running it for smaller set of batches for 2 epochs. 
- Train the model for 30K image set on GPU machine in runpod.io and retrieve the trained model, loss graph. 

## Customization

### Adding New Datasets
1. Modify the data paths in the script
2. Ensure captions are in the same format as Flickr8k
3. Adjust vocabulary frequency threshold if needed

### Model Modifications
- Change encoder: Modify `EncoderCNN` class
- Change decoder: Modify `DecoderRNN` class
- Add attention: Implement attention mechanism in decoder

## License

This implementation is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!
