# Multi-Dataset Image Captioning with PyTorch ResNet-LSTM/Attention

This project implements an image captioning model using ResNet50 as the image encoder and LSTM/LSTM+Attention as the text decoder, supporting both Flickr8k and Flickr30k datasets with advanced learning rate scheduling and attention visualization.

## 🚀 Quick Start - Key Training Scenarios

### 1. Train with Flickr8k + LSTM + Cosine LR Scheduling
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr8k \
    --decoder lstm \
    --epochs 15 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 15 \
    --min_lr 1e-6
```

### 2. Train with Flickr30k + Attention + Cosine LR Scheduling + Attention Visualization
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder attention \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 20 \
    --min_lr 1e-6 \
    --save_attention_viz \
    --attention_viz_samples 5 \
    --advanced_attention_viz
```

### 3. Train with Flickr8k + Attention + ReduceLROnPlateau + Basic Attention Visualization
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr8k \
    --decoder attention \
    --epochs 12 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --use_scheduler \
    --scheduler_type reduce_lr_on_plateau \
    --scheduler_patience 3 \
    --scheduler_factor 0.5 \
    --save_attention_viz \
    --attention_viz_samples 3
```

### 4. Train with Flickr30k + (without Attention) + Cosine LR Scheduling + 
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder lstm \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 20 \
    --min_lr 1e-6
```

### 5. Train with Flickr30k + Gpt2 decoder + Cosine LR Scheduling
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder gpt2 \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 20 \
    --min_lr 1e-6 
```

### 6. Train with Flickr30k + ViT encoder + Gpt2 decoder + Cosine LR Scheduling
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder gpt2 \
    --encoder vit \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 20 \
    --min_lr 1e-6 
```

### 7. Train with Flickr30k + ViT encoder + Gpt2 decoder + Cosine LR Scheduling [fine tuned]
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder gpt2 \
    --encoder vit \
    --num_workers 8 \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 20 \
    --min_lr 1e-6 
```

## 📋 Overview

The model architecture consists of:
- **Encoder**: ResNet50 (pretrained) for image feature extraction
- **Decoder**: LSTM or LSTM+Attention for generating captions
- **Supported Datasets**: Flickr8k (8,000 images) and Flickr30k (31,000 images)
- **Learning Rate Scheduling**: Multiple scheduler types for optimal training
- **Attention Visualization**: Visualize attention weights for attention-based models

## 🎯 Key Features

### 🔄 Learning Rate Scheduling
- **ReduceLROnPlateau**: Reduces LR when validation loss plateaus
- **CosineAnnealingLR**: Cosine annealing schedule (recommended)
- **StepLR**: Step-based decay
- **ExponentialLR**: Exponential decay
- **OneCycleLR**: 1cycle policy for super-convergence

### 👁️ Attention Visualization
- **Basic Visualization**: Simple attention heatmaps
- **Advanced Visualization**: Detailed attention analysis with multiple views
- **Heatmap Generation**: Save attention visualizations as PNG files
- **Sample Selection**: Control number of visualization samples

### 📊 Comprehensive Monitoring
- Real-time learning rate tracking
- Training/validation loss plots
- Learning rate over time visualization
- Attention weight visualization (for attention models)
- Automatic checkpoint saving and resuming

## 🛠️ Setup Instructions

### 1. Install Dependencies
Note: If venv needs to be installed
```bash
python3 -m venv venv
source venv/bin/activate
```
```bash
pip install -r requirements.txt
```

### 2. Download Datasets
```bash
# Automated download (recommended)
python download_datasets.py

# Or download specific datasets
python download_datasets.py --dataset flickr8k
python download_datasets.py --dataset flickr30k
```

### 3. Download Pretrained ResNet50
```bash
mkdir -p ../../resources/models
curl -k -o ../../resources/models/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## 🎮 Training Commands

### Basic Training (No Scheduling, No Visualization)
```bash
# Flickr8k with LSTM
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr8k \
    --decoder lstm \
    --epochs 10 \
    --batch_size 64

# Flickr30k with Attention
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder attention \
    --epochs 15 \
    --batch_size 32
```

### Advanced Training with Cosine LR Scheduling
```bash
# Flickr8k + LSTM + Cosine Annealing
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr8k \
    --decoder lstm \
    --epochs 15 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 15 \
    --min_lr 1e-6

# Flickr30k + Attention + Cosine Annealing
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder attention \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 20 \
    --min_lr 1e-6
```

### Training with Attention Visualization
```bash
# Basic attention visualization
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder attention \
    --epochs 15 \
    --batch_size 32 \
    --save_attention_viz \
    --attention_viz_samples 3

# Advanced attention visualization with cosine scheduling
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder attention \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 20 \
    --min_lr 1e-6 \
    --save_attention_viz \
    --attention_viz_samples 5 \
    --advanced_attention_viz
```

### Alternative Schedulers
```bash
# ReduceLROnPlateau (reduces LR when validation loss plateaus)
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr8k \
    --decoder attention \
    --epochs 15 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --use_scheduler \
    --scheduler_type reduce_lr_on_plateau \
    --scheduler_patience 3 \
    --scheduler_factor 0.5 \
    --save_attention_viz

# StepLR (reduces LR every N epochs)
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder lstm \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --use_scheduler \
    --scheduler_type step \
    --scheduler_step_size 5 \
    --scheduler_gamma 0.3
```

## 📊 Monitoring and Output

### Console Output Example
```
Epoch 5 completed (5/20):
  Training Loss: 2.34567
  Validation Loss: 2.12345
  Learning Rate: 0.000500
  Scheduler: CosineAnnealingLR (T_max=20)
  Model checkpoint saved (best: True)
```

### Generated Files
- `training_losses.png`: Training and validation loss plots
- `learning_rate.png`: Learning rate over time plot
- `attention_viz_sample_1.png`: Attention visualizations (if enabled)
- `loss_curve.json`: Training history with learning rates
- `metrics_history.json`: BLEU, WER, ROUGE metrics

### Attention Visualization Output
When using attention visualization, you'll get:
- **Basic Visualization**: Simple attention heatmaps overlaid on images
- **Advanced Visualization**: Multi-panel analysis with attention weights
- **Heatmap Files**: Saved as PNG files in the model directory

## 🔄 Resume Training

### Resume with Scheduler State
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --model_path saved_models/flickr30k/resnet_attention/model_epoch_10.pt \
    --epochs 5 \
    --use_scheduler \
    --scheduler_type cosine
```

### Resume with Attention Visualization
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --model_path saved_models/flickr30k/resnet_attention/model_epoch_15.pt \
    --epochs 5 \
    --save_attention_viz \
    --attention_viz_samples 3
```

## 🎛️ Command Line Arguments

### Basic Arguments
- `--dataset_type`: `flickr8k` or `flickr30k` (default: `flickr30k`)
- `--decoder`: `lstm` or `attention` (default: `lstm`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Training batch size (default: 128)
- `--learning_rate`: Initial learning rate (default: 0.0001)

### Learning Rate Scheduling
- `--use_scheduler`: Enable learning rate scheduling
- `--scheduler_type`: `reduce_lr_on_plateau`, `step`, `cosine`, `exponential`, `one_cycle`
- `--scheduler_patience`: Patience for ReduceLROnPlateau (default: 3)
- `--scheduler_factor`: Factor for reducing learning rate (default: 0.5)
- `--scheduler_step_size`: Step size for StepLR (default: 7)
- `--scheduler_gamma`: Gamma for StepLR/ExponentialLR (default: 0.1)
- `--scheduler_t_max`: T_max for CosineAnnealingLR (default: 10)
- `--min_lr`: Minimum learning rate (default: 1e-7)
- `--warmup_epochs`: Warmup epochs for OneCycleLR (default: 0)

### Attention Visualization
- `--save_attention_viz`: Save attention visualizations (attention decoder only)
- `--attention_viz_samples`: Number of attention visualization samples (default: 3)
- `--advanced_attention_viz`: Create advanced attention visualizations with heatmaps

### Model and Data
- `--embed_size`: Embedding size (default: 512)
- `--hidden_size`: LSTM hidden size (default: 1024)
- `--num_layers`: Number of LSTM layers (default: 3)
- `--attention_dim`: Attention dimension (default: 512)
- `--drop_prob`: Dropout probability (default: 0.3)
- `--num_workers`: Number of data loading workers (default: 16)

## 📈 Performance Recommendations

### Dataset-Specific Settings

#### Flickr8k (Faster Training)
```bash
# Recommended settings for Flickr8k
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr8k \
    --decoder lstm \
    --epochs 15 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 15
```

#### Flickr30k (Better Results)
```bash
# Recommended settings for Flickr30k
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder attention \
    --epochs 20 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 20 \
    --save_attention_viz \
    --attention_viz_samples 5
```

### Hardware-Specific Settings

#### GPU Training (Recommended)
```bash
# High-end GPU (16GB+ VRAM)
--batch_size 128 --num_workers 16

# Mid-range GPU (8GB VRAM)
--batch_size 64 --num_workers 8

# Entry-level GPU (4GB VRAM)
--batch_size 32 --num_workers 4
```

#### CPU Training
```bash
# CPU training (slower but works)
--batch_size 16 --num_workers 2
```

## 🔍 Testing and Evaluation

### Test Trained Model
```bash
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --mode test \
    --model_path saved_models/flickr30k/resnet_attention/best.pt \
    --dataset_type flickr30k
```

### Test Dataset Loading
```bash
python test_dataset_support.py
```

### Test Learning Rate Schedulers
```bash
python test_lr_scheduling.py
```

## 🏗️ Model Architecture

### Encoder (ResNet50)
- Extracts 2048-dimensional features from images
- Projects to configurable embedding space (default: 512)
- Parameters are frozen during training

### Decoder Options

#### LSTM Decoder
- Standard LSTM for sequence generation
- Faster training, good baseline performance
- Use with `--decoder lstm`

#### LSTM+Attention Decoder
- Attention mechanism over image features
- Better performance, attention visualization available
- Use with `--decoder attention`
- Enables attention visualization with `--save_attention_viz`

### Vocabulary
- Built from training captions with frequency threshold
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Flickr8k: ~3000-4000 words
- Flickr30k: ~8000-10000 words

## 📁 File Structure

```
src/image_caption_generation/
├── README.md                                           # This comprehensive guide
├── flickr_image_caption_with_pytorch_resnet_lstm.py   # Main training script
├── download_datasets.py                               # Dataset download script
├── test_dataset_support.py                            # Dataset testing script
├── test_lr_scheduling.py                              # LR scheduler testing
├── test_unique_directories.py                         # Unique directory testing
├── attention_visualization.py                         # Attention visualization utilities
├── training_utils.py                                  # Training utilities
├── metrics_utils.py                                   # Metrics computation
├── experiment_utils.py                                # Experiment management
├── requirements.txt                                   # Python dependencies
├── setup.sh                                           # Setup script
└── saved_models/                                      # Trained models and results
    ├── flickr8k/
    │   └── resnet_lstm/
    └── flickr30k/
        └── resnet_attention/
```

## 📊 Learning Rate Scheduling

### 🎯 **Supported Schedulers**
- **ReduceLROnPlateau**: Reduces LR when validation loss plateaus
- **StepLR**: Reduces LR every N epochs
- **CosineAnnealingLR**: Cosine annealing schedule
- **ExponentialLR**: Exponential decay
- **OneCycleLR**: One-cycle policy with warmup

### 🚀 **Quick Start with Schedulers**

```bash
# Cosine annealing scheduler (recommended)
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr8k \
    --decoder lstm \
    --epochs 15 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --use_scheduler \
    --scheduler_type cosine \
    --scheduler_t_max 15

# ReduceLROnPlateau with custom patience
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr30k \
    --decoder attention \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --use_scheduler \
    --scheduler_type reduce_lr_on_plateau \
    --scheduler_patience 5 \
    --scheduler_factor 0.5
```

### 📈 **Scheduler Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_scheduler` | False | Enable learning rate scheduling |
| `--scheduler_type` | reduce_lr_on_plateau | Scheduler type |
| `--scheduler_patience` | 3 | Patience for ReduceLROnPlateau |
| `--scheduler_factor` | 0.5 | Factor for reducing LR |
| `--scheduler_step_size` | 7 | Step size for StepLR |
| `--scheduler_gamma` | 0.1 | Gamma for StepLR/ExponentialLR |
| `--scheduler_t_max` | 10 | T_max for CosineAnnealingLR |
| `--min_lr` | 1e-7 | Minimum learning rate |
| `--warmup_epochs` | 0 | Warmup epochs for OneCycleLR |

### 📊 **Learning Rate Monitoring**
- **Real-time tracking**: Learning rate changes logged during training
- **Visualization**: Learning rate plots saved automatically
- **Scheduler info**: Detailed scheduler parameters displayed

## 🔧 Unique Run Directories & Parameter Tracking

### 🎯 **Automatic Parameter Tracking**
Each training run creates a unique directory based on all parameters used, ensuring:
- **No overwriting**: Different parameter combinations get separate directories
- **Complete reproducibility**: All parameters are saved and can be reproduced
- **Easy comparison**: Compare different runs with different hyperparameters

### 📊 **Generated Files for Each Run**

#### Configuration Files
- **`run_config.json`**: Complete parameter configuration in JSON format
- **`run_summary.txt`**: Human-readable summary with command to reproduce

#### Training Results
- **`training_losses.png`**: Training and validation loss plots
- **`learning_rate.png`**: Learning rate over time visualization
- **`loss_curve.json`**: Training history with learning rates
- **`metrics_history.json`**: BLEU, WER, ROUGE metrics
- **`best.pt`**: Best model checkpoint
- **`model_epoch_*.pt`**: Individual epoch checkpoints

#### Attention Visualizations (if enabled)
- **`attention_viz_sample_*.png`**: Attention heatmaps
- **`advanced_attention_viz_sample_*.png`**: Advanced attention analysis

### 🏷️ **Directory Naming Convention**
```
{dataset_type}_{decoder_type}{scheduler_suffix}{attention_suffix}_{timestamp}
```

**Examples:**
- `flickr8k_lstm_ep15_bs64_lr1e-03_cosine_20241201_143022` - Flickr8k + LSTM + 15 epochs + 64 batch size + 0.001 LR + Cosine scheduler
- `flickr30k_attention_ep30_bs128_lr1e-04_att_20241201_143045` - Flickr30k + Attention + 30 epochs + 128 batch size + 0.0001 LR + Attention viz

### 📝 **Example Configuration Files**

#### `run_config.json`
```json
{
  "dataset_type": "flickr30k",
  "decoder_type": "attention",
  "epochs": 20,
  "batch_size": 32,
  "learning_rate": 0.0005,
  "embed_size": 512,
  "hidden_size": 1024,
  "num_layers": 3,
  "attention_dim": 512,
  "drop_prob": 0.3,
  "num_workers": 4,
  "use_scheduler": true,
  "scheduler_type": "cosine",
  "scheduler_t_max": 20,
  "min_lr": 1e-6,
  "save_attention_viz": true,
  "attention_viz_samples": 5,
  "advanced_attention_viz": true,
  "timestamp": "20241201_143022"
}
```

#### `run_summary.txt`
```
Training Run Summary
===================

Run Directory: flickr30k_attention_ep20_bs32_lr5e-04_cosine_att_20241201_143022
Timestamp: 20241201_143022

Dataset: flickr30k
Decoder: attention
Epochs: 20
Batch Size: 32
Learning Rate: 0.0005
Embed Size: 512
Hidden Size: 1024
Num Layers: 3
Attention Dim: 512
Dropout: 0.3
Num Workers: 4

Learning Rate Scheduler: cosine
  T_max: 20
  Min LR: 1e-06

Attention Visualization: Enabled
  Samples: 5
  Advanced: True

Full configuration saved to: run_config.json
```

### 🧪 **Testing Unique Directory Creation**
Test the unique directory functionality:
```bash
python test_unique_directories.py
```

This will create test directories with different parameter combinations and show you the generated files.

### 🔄 **Resuming Training with Unique Directories**
When resuming training, the script will:
1. Load the model from the checkpoint
2. Create a new unique directory for the resumed run
3. Save all new training data in the new directory
4. Preserve the original run's data

This ensures you can compare the original run with the resumed run.

## 🚨 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size
   --batch_size 32 --num_workers 4
   ```

2. **Learning rate too small too early**
   ```bash
   # Increase patience or factor
   --scheduler_patience 5 --scheduler_factor 0.7
   ```

3. **Attention visualization not working**
   ```bash
   # Ensure using attention decoder
   --decoder attention --save_attention_viz
   ```

4. **Dataset not found**
   ```bash
   # Download datasets
   python download_datasets.py
   ```

5. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Performance Tips

- **Use Cosine Annealing**: Best overall performance for most cases
- **Attention Decoder**: Better results but slower training
- **Flickr30k**: Better results but longer training time
- **GPU Training**: Significantly faster than CPU
- **Batch Size**: Larger batches generally better (if memory allows)

## 📊 Expected Performance

### Training Time Estimates
- **Flickr8k + LSTM**: ~30 mins for 15 epochs (GPU)
- **Flickr8k + Attention**: ~45 mins for 15 epochs (GPU)
- **Flickr30k + LSTM**: ~2 hours for 20 epochs (GPU)
- **Flickr30k + Attention**: ~3 hours for 20 epochs (GPU)

### Typical Metrics
- **Loss**: Starts ~4.0, converges to ~2.0-2.5
- **BLEU Score**: 0.2-0.4 (varies by dataset and decoder)
- **Memory Usage**: 4-8 GB GPU memory with recommended settings

## 🎯 Best Practices

### 1. Start Simple
```bash
# Start with basic LSTM + Flickr8k
python flickr_image_caption_with_pytorch_resnet_lstm.py \
    --dataset_type flickr8k \
    --decoder lstm \
    --epochs 10
```

### 2. Add Learning Rate Scheduling
```bash
# Add cosine scheduling for better convergence
--use_scheduler --scheduler_type cosine --scheduler_t_max 10
```

### 3. Upgrade to Attention
```bash
# Switch to attention decoder for better performance
--decoder attention --save_attention_viz
```

### 4. Scale to Flickr30k
```bash
# Use larger dataset for better results
--dataset_type flickr30k --epochs 20
```

## 📝 License

This implementation is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests! 