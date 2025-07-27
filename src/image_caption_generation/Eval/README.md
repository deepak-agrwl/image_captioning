# Image Caption Generation Model Evaluation

This directory contains the evaluation functionality for trained image captioning models.

## Quickstart

### Evaluate Any model trained on Flickr30k
Verfied with resnet+lstm and vit+gpt2 models

```bash
python eval.py \
    --model_path "/path/to/your/model.pth" \
    --dataset_type flickr30k \
    --batch_size 32 \
    --num_workers 4 \
    --device auto \
    --output_dir ./eval_results
    --eval_fraction 0.2
```

## Files

- `eval.py` - Main evaluation script with command line interface
- `demo_eval.py` - Demonstration script showing how to use the evaluation
- `README.md` - This documentation file

## Features

The evaluation script provides comprehensive metrics for image captioning models:

### Metrics Computed

1. **BLEU-4 Score** - Standard metric for evaluating caption quality
2. **WER (Word Error Rate)** - Measures word-level differences 
3. **ROUGE-L Score** - Measures longest common subsequence similarity
4. **Cosine Similarity** - Semantic similarity using sentence embeddings

### Supported Models

- Models trained with the main training script (`flickr_image_caption_with_pytorch_resnet_lstm.py`)
- Supports both LSTM and Attention-based decoders
- Works with both ResNet and ViT encoders

### Supported Datasets

- Flickr8k
- Flickr30k

## Usage

### Basic Usage

```bash
cd src/image_caption_generation/Eval
python eval.py --model_path /path/to/your/model.pth --dataset_type flickr30k
```

### Full Options

```bash
python eval.py \
    --model_path /path/to/your/model.pth \
    --dataset_type flickr30k \
    --batch_size 32 \
    --num_workers 4 \
    --max_samples 1000 \
    --device auto \
    --output_dir ./results \
    --no_cosine_similarity
```

### Command Line Arguments

- `--model_path` (required): Path to the trained model checkpoint (.pth file)
- `--dataset_type`: Dataset to evaluate on (`flickr8k` or `flickr30k`, default: `flickr30k`)
- `--batch_size`: Batch size for evaluation (default: `32`)
- `--num_workers`: Number of workers for data loading (default: `4`)
- `--max_samples`: Maximum number of samples to evaluate (default: all)
- `--device`: Device to use (`auto`, `cpu`, `cuda`, `mps`, default: `auto`)
- `--output_dir`: Directory to save detailed results (optional)
- `--no_cosine_similarity`: Skip cosine similarity computation

## Example Usage

### Quick Evaluation

Evaluate a model on 100 samples:

```bash
python eval.py \
    --model_path ../saved_models/best_model.pth \
    --dataset_type flickr30k \
    --max_samples 100
```

### Full Evaluation with Results Saved

```bash
python eval.py \
    --model_path ../saved_models/best_model.pth \
    --dataset_type flickr30k \
    --batch_size 16 \
    --output_dir ./evaluation_results_$(date +%Y%m%d_%H%M%S)
```

### Demo

Run the demonstration script:

```bash
python demo_eval.py
```

## Output

### Console Output

The script prints:
- Model configuration details
- Evaluation progress
- Final metrics summary

Example output:
```
============================================================
EVALUATION RESULTS
============================================================
Model: best_model.pth
Dataset: flickr30k
Samples evaluated: 1000
Decoder type: attention
Encoder type: resnet
----------------------------------------
METRICS:
  BLEU-4:        0.2534
  WER:           0.5432
  ROUGE-L:       0.4123
  Cosine Sim:    0.7654 ± 0.1234
============================================================
```

### Output Files (if --output_dir specified)

1. **evaluation_summary.json** - Complete evaluation summary with metrics and configuration
2. **detailed_results.csv** - Detailed results with generated and reference captions
3. **detailed_results.json** - Same as CSV but in JSON format

## Requirements

The evaluation script requires the following dependencies:

```
torch
torchvision
transformers
scikit-learn
pandas
numpy
nltk
jiwer
rouge-score
tqdm
```

These are included in the main project requirements.

## Cosine Similarity Details

The cosine similarity computation uses sentence embeddings:

1. **Primary Method**: Uses sentence-transformers model (`sentence-transformers/all-MiniLM-L6-v2`)
2. **Fallback Method**: Simple word-based embeddings if sentence-transformers is not available

The cosine similarity computes semantic similarity between generated captions and reference captions, taking the maximum similarity when multiple references are available.

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure the model path is correct and the model was saved with the training script
2. **Dataset Not Found**: Check that the dataset paths in `DATASET_CONFIGS` are correct
3. **Memory Issues**: Reduce `batch_size` or `max_samples` for evaluation
4. **Slow Evaluation**: Use GPU (`--device cuda`) if available

### Performance Tips

- Use GPU for faster evaluation: `--device cuda`
- Reduce batch size if encountering memory issues
- Use `--max_samples` for quick testing
- Increase `--num_workers` for faster data loading (but may use more RAM)

## Integration with Main Training Script

The evaluation script seamlessly integrates with models trained using the main training script. It automatically:

- Loads the correct model architecture based on saved hyperparameters
- Uses the saved vocabulary
- Handles different decoder types (LSTM, Attention, GPT2)
- Works with different encoder types (ResNet, ViT)

## Example with Provided Model

To evaluate the example model mentioned in the request:

```bash
python eval.py \
    --model_path /Users/deagrawa/github/image_captioning/resources/output/flickr30k_attention_ep25_bs128_lr1e4_cosine_att_20250712_164848/flickr30k/resnet_attention/best_model.pth \
    --dataset_type flickr30k \
    --batch_size 16 \
    --max_samples 500 \
    --output_dir ./attention_model_eval_results
```

This will evaluate the attention-based model on 500 Flickr30k samples and save detailed results. 